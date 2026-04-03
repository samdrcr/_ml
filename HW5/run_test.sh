#!/bin/bash
export AGENT_DIR="$HOME/.nexus_agent"

echo "Initializing environment..."
rm -rf ./test_build
mkdir -p ./test_build

# Feeding the instruction directly to the python script
echo "Generating API boilerplate..."
python3 agent_core.py <<EOF
Create a Python FastAPI app in ./test_build/main.py with a GET root endpoint returning {"status": "active"}. 
Then exit.
quit
EOF

if [ -f "./test_build/main.py" ]; then
    echo "Success: File created."
    cat ./test_build/main.py
else
    echo "Failure: File not found."
    exit 1
fi