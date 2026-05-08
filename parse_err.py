import io
import sys

with io.open('error_log.txt', 'r', encoding='utf-16le', errors='ignore') as f:
    text = f.read()

# Just print the last 30 lines which has the traceback
lines = text.split('\n')
for line in lines[-30:]:
    print(line)
