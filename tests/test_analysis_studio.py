import unittest
import numpy as np
import httpx
from analysis_engine import Connect4AnalysisEngine, classify_move_quality, detect_tactical_threats
from web_server import app


class TestAlphaGoStudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Connect4AnalysisEngine(default_simulations=64)

    def test_classify_move_quality(self):
        label, color = classify_move_quality(0.5)
        self.assertEqual(label, "Best Move")

        label, color = classify_move_quality(5.0)
        self.assertEqual(label, "Good")

        label, color = classify_move_quality(15.0)
        self.assertEqual(label, "Mistake")

        label, color = classify_move_quality(25.0)
        self.assertEqual(label, "Blunder")

    def test_tactical_threats(self):
        board = np.zeros((6, 7), dtype=np.int8)
        # Create 3-in-a-row for player 1
        board[0, 0] = 1
        board[0, 1] = 1
        board[0, 2] = 1
        threats = detect_tactical_threats(board, to_play=1)
        self.assertIn(3, threats.immediate_wins)

    def test_analysis_engine(self):
        board = np.zeros((6, 7), dtype=np.int8)
        res = self.engine.analyze(board, to_play=1, simulations=64)
        self.assertEqual(len(res.candidates), 7)
        self.assertIsNotNone(res.best_move)
        self.assertTrue(0 <= res.win_rate_red <= 100)
        self.assertTrue(0 <= res.win_rate_yellow <= 100)
        self.assertGreater(len(res.principal_variation), 0)

    def test_api_routes(self):
        from starlette.testclient import TestClient
        client = TestClient(app)
        r = client.get("/api/state")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("board", data)
        self.assertIn("analysis", data)


if __name__ == "__main__":
    unittest.main()
