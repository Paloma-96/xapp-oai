import logging
import  xapp_control
import xapp_control_ricbypass
from  ran_messages_pb2 import *
from time import sleep
BYPASS_RIC = True
import sqlite3
import datetime
import pandas as pd


def db_init():
    # Create a connection to the SQLite database file (or create a new one if it doesn't exist)
    conn = sqlite3.connect('toa_measurements.db')

    # Create a cursor for executing SQL queries
    c = conn.cursor()

    # Create the table (if it doesn't exist)
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
               (gnb_id TEXT, rnti INTEGER, toa_val REAL, snr REAL, timestamp TIMESTAMP)''')
    
    return conn, c

    

localIP = "127.0.0.1"
remoteIP = "10.75.10.77"

GNB_ID = 1
TOA_LIST = 4

# Convert protobuf data to InfluxDB line protocol format
def convert_protobuf_to_sqlite(protobuf_data):
    gnb_id = None
    toa_val = None
    snr = None
    rnti = None
    for param in protobuf_data:
        print("PARAM--------------------------\n")
        print(param.key)
        if param.key == GNB_ID:
            print("GNB ID--------------------------\n")
            print("GNB_ID: ", param.string_value)
            gnb_id = param.string_value

        elif param.key == TOA_LIST:
            print("TOA LIST--------------------------\n")
            print("TOA_LIST: ", param.toa)
            
            print("VARIE--------------------------\n")
            print("TOA_VAL: ", param.toa.toa_val)
            print("SNR: ", param.toa.snr)
            print("RNTI: ", param.toa.rnti)
            toa_val = param.toa.toa_val
            snr = param.toa.snr
            rnti = param.toa.rnti
    return gnb_id, rnti, toa_val, snr

def main():
    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    conn,c = db_init()
    
    if BYPASS_RIC: # connect directly to gnb_emu
        #xapp_control_ricbypass.receive_from_socket()
        print("Encoding initial ric indication request")
        master_mess = RAN_message()
        master_mess.msg_type = RAN_message_type.INDICATION_REQUEST
        inner_mess = RAN_indication_request()
        inner_mess.target_params.extend([RAN_parameter.GNB_ID, RAN_parameter.TOA_LIST])
        #inner_mess.target_params.extend([RAN_parameter.GNB_ID, RAN_parameter.UE_LIST])
        #inner_mess.target_params.extend([RAN_parameter.GNB_ID])
        master_mess.ran_indication_request.CopyFrom(inner_mess)
        buf = master_mess.SerializeToString()
        xapp_control_ricbypass.send_to_socket(buf, remoteIP)
        xapp_control_ricbypass.send_to_socket(buf, localIP)
        print("request sent, now waiting for incoming answers")

        while True:
            r_buf = xapp_control_ricbypass.receive_from_socket()
            # PALOMA HACK qui si implementa la logica di ricezione delle risposte
            ran_ind_resp = RAN_indication_response()
            ran_ind_resp.ParseFromString(r_buf)
            #print(ran_ind_resp)
            sleep(1)
            '''
            print("[PALOMA HACK]: \n")
            print("--------------------------------------------------\n")
            print("[PALOMA HACK]: ran_ind_resp.param_map[0]\n")
            print(ran_ind_resp.param_map[0])
            print("--------------------------------------------------\n")
            print("[PALOMA HACK]: ran_ind_resp.param_map[1]\n")
            print(ran_ind_resp.param_map[1])
            print("--------------------------------------------------\n")
            print("[PALOMA HACK]: ran_ind_resp.param_map[1].toa\n")
            print(ran_ind_resp.param_map[1].toa)
            print("--------------------------------------------------\n")
            print("[PALOMA HACK]: ran_ind_resp.param_map[1].toa.snr\n")
            print(ran_ind_resp.param_map[1].toa.snr)
            print("--------------------------------------------------\n")
            '''

            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            
            gnb_id, rnti, toa_val, snr = convert_protobuf_to_sqlite(ran_ind_resp.param_map)

            if (pd.isna(toa_val)):
                print("TOA is NaN, skipping this measurement")
                continue
            
            # Insert data into the table
            c.execute("INSERT INTO measurements (gnb_id, rnti, toa_val, snr, timestamp) VALUES (?, ?, ?, ?, ?)",
                    (gnb_id, rnti, toa_val, snr, date))
            # Commit the changes and close the connection
            conn.commit()
            #conn.close()

            xapp_control_ricbypass.send_to_socket(buf, remoteIP)
            xapp_control_ricbypass.send_to_socket(buf, localIP)

        r_buf = xapp_control_ricbypass.receive_from_socket()
        ran_ind_resp = RAN_indication_response()
        ran_ind_resp.ParseFromString(r_buf)
        print(ran_ind_resp)

        exit()
        
    # else: # connect to RIC
    control_sck = xapp_control.open_control_socket(4200)

    while True:
        logging.info("loop again")
        data_sck = xapp_control.receive_from_socket(control_sck)
        if len(data_sck) <= 0:
            logging.info("leq 0 data")
            if len(data_sck) == 0:
                continue
            else:
                logging.info('Negative value for socket')
                break
        else:
            logging.info('Received data: ' + repr(data_sck))
            logging.info("Sending something back")
            xapp_control.send_socket(control_sck, "test test test")


if __name__ == '__main__':
    main()

