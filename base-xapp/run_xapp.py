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
from pymongo import MongoClient
from google.protobuf.json_format import MessageToJson
import json


from threading import Event


def db_init():
    # Create a connection to the SQLite database file (or create a new one if it doesn't exist)
    conn = sqlite3.connect('toa_measurements.db', check_same_thread=False)

    # Create a cursor for executing SQL queries
    c = conn.cursor()

    # Create a table (if it doesn't exist) 

    #ask user for gnb id
    gnb_id = input("Enter gNB ID: ")

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    # Modify the CREATE TABLE command
    c.execute(f'''CREATE TABLE IF NOT EXISTS measurements_{current_datetime}
                    (gnb_id text, rnti text, toa_val_0 real, toa_val_1 real, toa_val_2 real, snr_0 real, snr_1 real, snr_2 real, timestamp text)''')
    
    return conn, c, current_datetime, gnb_id

def mongodb_init():
    client = MongoClient('mongodb://root:rootpassword@localhost:27017/')
    db = client['toa_measurements']
    collection = db['toa_measurements']
    return collection

def get_data_from_gnb(ip, conn, c, current_datetime, event):
    print("Encoding initial ric indication request")
    master_mess = RAN_message()
    master_mess.msg_type = RAN_message_type.INDICATION_REQUEST
    inner_mess = RAN_indication_request()
    inner_mess.target_params.extend([RAN_parameter.GNB_ID, RAN_parameter.TOA_LIST])
    master_mess.ran_indication_request.CopyFrom(inner_mess)
    buf = master_mess.SerializeToString()

    while True:
        xapp_control_ricbypass.send_to_socket(buf, ip)
        #print(f"Request sent to {ip}, now waiting for incoming answer")
        try:
            r_buf = xapp_control_ricbypass.receive_from_socket(timeout=1)
            #print(f"Received data from {ip}: {r_buf}")
            ran_ind_resp = RAN_indication_response()
            ran_ind_resp.ParseFromString(r_buf)
            #print(ran_ind_resp)

            date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            gnb_id, rnti, toa_val, snr = convert_protobuf_to_sqlite(ran_ind_resp.param_map)

            #print(f"Received data from {ip}: gnb_id: {gnb_id}, rnti: {rnti}, toa_val: {toa_val}, snr: {snr}, timestamp: {date}")

            query = f"INSERT INTO measurements_{current_datetime} (gnb_id, rnti, toa_val_0, toa_val_1, toa_val_2, snr_0, snr_1, snr_2, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            c.execute(query, (gnb_id, rnti, toa_val[0], toa_val[1], toa_val[2], snr[0], snr[1], snr[2], date))

            conn.commit()

            
        except Exception as e:
            print(f"Error receiving data from {ip}: {e}")
        if event.is_set():
            break
        time.sleep(0.1)
  

localIP = "127.0.0.1"
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

# Add this function to get user input and check if it's 'n'
def get_user_input():
    user_input = input("Press 'n' to switch to next gNB or 'q' to quit: ")
    return user_input

#def main():
    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    conn, c, current_datetime, current_gnb = db_init()

    event = Event()

    print("Starting xApp")
    print("Current datetime: ", current_datetime)

    gnb_ips = ["127.0.0.1", "10.75.10.77", "10.75.10.1"]

    interactive = False

    if BYPASS_RIC:  # connect directly to gnb_emu
        while True:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                event.clear()

                executor.submit(get_data_from_gnb, gnb_ips[int(current_gnb)], conn, c, current_datetime, event)

                if (interactive):
                    sleep(5)  # Adjust the sleep duration as needed
                    user_input = get_user_input()

                    if user_input == 'n':
                        event.set()
                        current_gnb = (current_gnb + 1) % len(gnb_ips)
                        if current_gnb == 0:
                            break
                        print("Switching to gNB: ", gnb_ips[current_gnb])
                    if user_input == 'q':
                        event.set()
                        break
                    else:
                        print("Continuing processing gNB: ", gnb_ips[current_gnb])
                else:
                    # sleep 60 seconds but print out the current % of the time every 10 seconds
                    for i in range(0, 120, 10):
                        print(f"Processing - {i/120*100}%")
                        time.sleep(10)

                    event.set()
                    break

                
        
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

def main():
    # configure logger and console output
    logging.basicConfig(level=logging.DEBUG, filename='/home/xapp-logger.log', filemode='a+',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    collection = mongodb_init()
    
    if BYPASS_RIC: # connect directly to gnb_emu
        #xapp_control_ricbypass.receive_from_socket()
        print("Encoding initial ric indication request")
        master_mess = RAN_message()
        master_mess.msg_type = RAN_message_type.INDICATION_REQUEST
        inner_mess = RAN_indication_request()
        inner_mess.target_params.extend([RAN_parameter.GNB_ID, RAN_parameter.TOA_LIST])
        #inner_mess.target_params.extend([RAN_parameter.GNB_ID])
        master_mess.ran_indication_request.CopyFrom(inner_mess)
        buf = master_mess.SerializeToString()
        xapp_control_ricbypass.send_to_socket(buf, localIP)
        print("request sent, now waiting for incoming answers")

        while True:
            r_buf = xapp_control_ricbypass.receive_from_socket()
            print("Received data: ", r_buf)
            ran_ind_resp = RAN_indication_response()
            ran_ind_resp.ParseFromString(r_buf)
            json_message = MessageToJson(ran_ind_resp)
            # add to the json message the current datetime
            json_message = json_message[:-1] + ', "timestamp": "' + datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f") + '"}'
            #print("json message: ", json_message)
            
            json_message_object = json.loads(json_message)
                
            # Inserting the loaded data in the Collection
            # if JSON contains data more than one entry
            # insert_many is used else insert_one is used
            if isinstance(json_message_object, list):
                collection.insert_many(json_message_object) 
            else:
                collection.insert_one(json_message_object)

            print(ran_ind_resp)

            sleep(0.3)
            xapp_control_ricbypass.send_to_socket(buf, localIP)

        r_buf = xapp_control_ricbypass.receive_from_socket()
        ran_ind_resp = RAN_indication_response()
        ran_ind_resp.ParseFromString(r_buf)
        print(ran_ind_resp)

        exit()

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

