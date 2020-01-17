import sys
import thread
import webbrowser
import time
import BaseHTTPServer, SimpleHTTPServer
from shutil import copyfile
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def start_server(port):
    httpd = BaseHTTPServer.HTTPServer(('localhost', port), SimpleHTTPServer.SimpleHTTPRequestHandler)
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", type=int, metavar='<int>', default=9000,
                        help="Port used to display the visualization")

    args = parser.parse_args()

    assert 65535 >= args.port >= 1024, "Choose a port that is >= 1024 and <= 65535"

    thread.start_new_thread(start_server, (args.port, ))
    url = 'http://localhost:' + str(args.port)
    webbrowser.open_new(url)
    logger.info('Visualization ready on %s', url)


    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()


