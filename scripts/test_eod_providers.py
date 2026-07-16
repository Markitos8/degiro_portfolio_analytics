#!/usr/bin/env python
"""Lightweight unit tests for EOD provider helpers (no live paid APIs required)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from degiro_analytics import market_data as md


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class EodProviderTests(unittest.TestCase):
    def test_eodhd_symbol_mapping(self):
        self.assertEqual(md._yahoo_to_eodhd_symbol("AAPL"), "AAPL.US")
        self.assertEqual(md._yahoo_to_eodhd_symbol("IWDA.AS"), "IWDA.AS")
        self.assertEqual(md._yahoo_to_eodhd_symbol("VWCE.DE"), "VWCE.XETRA")
        self.assertEqual(md._yahoo_to_eodhd_symbol("^GSPC"), "GSPC.INDX")
        self.assertEqual(md._yahoo_to_eodhd_symbol("EURUSD=X"), "EURUSD.FOREX")

    def test_polygon_symbol_mapping(self):
        self.assertEqual(md._yahoo_to_polygon_ticker("AAPL"), "AAPL")
        self.assertEqual(md._yahoo_to_polygon_ticker("^GSPC"), "I:SPX")
        self.assertEqual(md._yahoo_to_polygon_ticker("EURUSD=X"), "C:EURUSD")
        self.assertIsNone(md._yahoo_to_polygon_ticker("IWDA.AS"))
        self.assertIsNone(md._yahoo_to_polygon_ticker("VWCE.DE"))

    def test_polygon_parser(self):
        md.POLYGON_API_KEY = "test-key"
        payload = {
            "status": "OK",
            "results": [
                {"t": 1704153600000, "c": 185.64},
                {"t": 1704240000000, "c": 184.25},
            ],
        }
        with patch("degiro_analytics.market_data.requests.get", return_value=FakeResponse(payload)):
            series, source = md._fetch_polygon("AAPL", "2024-01-02", "2024-01-03")
        self.assertEqual(source, "polygon")
        self.assertIsNotNone(series)
        self.assertEqual(len(series), 2)
        self.assertAlmostEqual(float(series.iloc[0]), 185.64)

    def test_eodhd_parser(self):
        md.EODHD_API_KEY = "test-key"
        payload = [
            {"date": "2024-01-02", "close": 185.64, "adjusted_close": 183.56},
            {"date": "2024-01-03", "close": 184.25, "adjusted_close": 182.18},
        ]
        with patch("degiro_analytics.market_data.requests.get", return_value=FakeResponse(payload)):
            series, source = md._fetch_eodhd("AAPL", "2024-01-02", "2024-01-03")
        self.assertEqual(source, "eodhd")
        self.assertEqual(len(series), 2)
        self.assertAlmostEqual(float(series.iloc[0]), 183.56)

    def test_provider_order(self):
        names = [name for name, _ in md._PROVIDER_FETCHERS]
        self.assertEqual(names, ["eodhd", "polygon", "yahoo_chart", "yfinance"])


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))
