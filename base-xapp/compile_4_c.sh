#!/bin/bash
protoc -I=. --c_out=. ./oai-oran-protolib/ran_messages.proto
