from enum import Enum


class MessageType(str, Enum):
    ADD_REQUEST = "add_request"
    ABORT_REQUEST = "abort_request"
    TOKEN = "token"
    FINISHED = "finished"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SHUTDOWN = "shutdown"
    SHUTDOWN_ACK = "shutdown_ack"


TERMINAL_MESSAGE_TYPES = {
    MessageType.FINISHED,
    MessageType.ERROR,
}
