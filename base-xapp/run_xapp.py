import logging
import  xapp_control
import xapp_control_ricbypass
from  ran_messages_pb2 import *
from time import sleep
BYPASS_RIC = True
import sqlite3
import datetime
import pandas as pd
import concurrent.futures
import time


def db_init():
    # Create a connection to the SQLite database file (or create a new one if it doesn't exist)
    conn = sqlite3.connect('toa_measurements.db', check_same_thread=False)

    # Create a cursor for executing SQL queries
    c = conn.cursor()

    # Create the table (if it doesn't exist)
    c.execute('''CREATE TABLE IF NOT EXISTS measurements
               (gnb_id TEXT, rnti INTEGER, toa_val REAL, snr REAL, timestamp TIMESTAMP)''')
    
    return conn, c

def get_data_from_gnb(ip, conn, c):
    print("Encoding initial ric indication request")
    master_mess = RAN_message()
    master_mess.msg_type = RAN_message_type.INDICATION_REQUEST
    inner_mess = RAN_indication_request()
    inner_mess.target_params.extend([RAN_parameter.GNB_ID, RAN_parameter.TOA_LIST])
    master_mess.ran_indication_request.CopyFrom(inner_mess)
    buf = master_mess.SerializeToString()

    while True:
        xapp_control_ricbypass.send_to_socket(buf, ip)
        print(f"Request sent to {ip}, now waiting for incoming answer")
        try:
            r_buf = xapp_control_ricbypass.receive_from_socket(timeout=1)
            ran_ind_resp = RAN_indication_response()
            ran_ind_resp.ParseFromString(r_buf)
            print(ran_ind_resp)

            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            gnb_id, rnti, toa_val, snr = convert_protobuf_to_sqlite(ran_ind_resp.param_map)

            c.execute("INSERT INTO measurements (gnb_id, rnti, toa_val, snr, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (gnb_id, rnti, toa_val, snr, date))
            conn.commit()
        except Exception as e:
            print(f"Error receiving data from {ip}: {e}")

        time.sleep(1)
  

#localIP = "127.0.0.1"
#remoteIP1 = "10.75.10.77"
#remoteIP2 = "10.75.10.1"

GNB_ID = 1
TOA_LIST = 4

# Convert protobuf data to InfluxDB line protocol format
def convert_protobuf_to_sqlite(protobuf_data):
    data = {}
    for param in protobuf_data:
        if param.key == GNB_ID:
            data['gnb_id'] = param.string_value
        elif param.key == TOA_LIST:
            data['toa_val'] = param.toa.toa_val
            data['snr'] = param.toa.snr
            data['rnti'] = param.toa.rnti
    return data.get('gnb_id', None), data.get('rnti', None), data.get('toa_val', None), data.get('snr', None)


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

    gnb_ips = ["127.0.0.1", "10.75.10.77", "10.75.10.1"]
    
    if BYPASS_RIC: # connect directly to gnb_emu
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gnb_ips)) as executor:
            futures = {executor.submit(get_data_from_gnb, ip, conn, c): ip for ip in gnb_ips}
        
    else: # connect to RIC
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

