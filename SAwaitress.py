from waitress import serve
import SAFlask

serve(SAFlask.app, port=8000, threads=6)