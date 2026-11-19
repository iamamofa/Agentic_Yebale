"""Launcher lifecycle checks without creating public tunnels in the test suite."""

import socket
import unittest
from unittest.mock import MagicMock

from start_app import available_port, stop, wait_for_server


class PublicLauncherTests(unittest.TestCase):
    def test_occupied_port_does_not_replace_an_existing_server(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', 0))
            occupied.listen()
            original = occupied.getsockname()[1]
            selected = available_port(original)
            self.assertNotEqual(original, selected)
            self.assertGreater(selected, 0)

    def test_failed_child_does_not_leave_startup_waiting(self):
        process = MagicMock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, 'stopped during startup'):
            wait_for_server(process, 'http://127.0.0.1:1')

    def test_shutdown_terminates_managed_process(self):
        process = MagicMock()
        process.poll.return_value = None
        stop(process)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=6)
