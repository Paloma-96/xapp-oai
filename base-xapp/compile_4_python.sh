#!/bin/bash
protoc -I=. --python_out=. ./oai-oran-protolib/ran_messages.proto
