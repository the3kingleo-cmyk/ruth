import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

from ruth.app.server import Life, make_handler


class TestApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.life = Life(tempfile.mkdtemp())
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.life))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        c.request(method, path, None if body is None else json.dumps(body), h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, data

    def test_page_and_assets_are_local(self):
        code, page = self.req("GET", "/")
        self.assertEqual(code, 200)
        self.assertNotIn(b"http://", page.replace(b"http://127.0.0.1", b""))
        self.assertNotIn(b"https://", page)
        for asset in ("/app.js", "/style.css", "/icon.svg"):
            self.assertEqual(self.req("GET", asset)[0], 200)

    def test_conversation_privacy_and_sleep(self):
        code, _ = self.req("POST", "/api/teach", {"text": "The sky is blue today. " * 3})
        self.assertEqual(code, 200)
        code, data = self.req("POST", "/api/talk", {"text": "The vault code is 7294.", "private": True})
        kept = json.loads(data)["kept"]
        self.assertTrue(4 <= kept <= 24)   # the information, not the familiar words
        code, data = self.req("POST", "/api/talk", {"text": "How is the sky?"})
        reply = json.loads(data)
        self.assertIn("text", reply)
        self.assertNotIn("7294", json.dumps(reply))
        code, data = self.req("POST", "/api/sleep", {})
        self.assertEqual(code, 200)
        self.assertIn("nightmares", json.loads(data))
        state = json.loads(self.req("GET", "/api/state")[1])
        self.assertEqual(len(state["activity"]), state["neurons"])

    def test_senses_and_voice(self):
        code, data = self.req("POST", "/api/hear", {"rate": 16000, "samples": [0.1] * 3200})
        self.assertEqual(json.loads(data)["moments"], 20)
        code, data = self.req("POST", "/api/see", {"w": 2, "h": 2, "pixels": [0, 1, 0, 1], "t": 0})
        self.assertEqual(code, 200)
        code, data = self.req("POST", "/api/voice", {"text": "hi"})
        self.assertGreater(len(json.loads(data)["samples"]), 0)

    def test_background_replay_runs_while_idle(self):
        import time
        life = Life(tempfile.mkdtemp())
        life.mind.teach("The sky is blue today. " * 4)
        life.touched = time.time() - 60
        t = threading.Thread(target=life.background, kwargs={"idle_replay": 1.0}, daemon=True)
        t.start()
        deadline = time.time() + 20
        while life.replayed == 0 and time.time() < deadline:
            time.sleep(0.2)
        life.stop.set()
        self.assertGreater(life.replayed, 0)

    def test_refuses_non_local_and_non_json(self):
        self.assertEqual(self.req("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)
        c = http.client.HTTPConnection("127.0.0.1", self.port)
        c.request("POST", "/api/talk", "text=hi", {"Content-Type": "text/plain"})
        self.assertEqual(c.getresponse().status, 415)
        self.assertEqual(self.req("GET", "/../../etc/passwd")[0], 404)


if __name__ == "__main__":
    unittest.main()
