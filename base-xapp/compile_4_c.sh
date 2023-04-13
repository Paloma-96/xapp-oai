#!/bin/bash
protoc -I=. --c_out=. ./ran_messages.proto
