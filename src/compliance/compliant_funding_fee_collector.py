
import logging
import asyncio
from .binance_api_rate_limiter import BinanceAPIRateLimiter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CompliantFundingFeeCollector:
    def __init__(self, binance_client, rate_limiter: BinanceAPIRateLimiter):
        self.client = binance_client
        self.rate_limiter = rate_limiter
        self.shutdown_event = asyncio.Event()
        self.monitoring_task = None

    async def start_monitoring(self, symbol):
        if self.monitoring_task:
            logging.warning("Monitoring is already running.")
            return
        self.monitoring_task = asyncio.create_task(self._monitor_funding_rate(symbol))

    async def _monitor_funding_rate(self, symbol):
        while not self.shutdown_event.is_set():
            if not await self.rate_limiter.wait_for_request_weight(5): # weight for premium index
                await asyncio.sleep(60)
                continue

            try:
                premium_index = await self.client.get_premium_index(symbol=symbol)
                funding_rate = float(premium_index['lastFundingRate'])
                logging.info(f"Current funding rate for {symbol}: {funding_rate}")
                
                # In a real bot, you would add logic here to act on the funding rate
                # e.g., if it's positive and you have a short, you collect fees.

            except Exception as e:
                logging.error(f"Error fetching funding rate: {e}")
            
            # Funding rates change every 8 hours, so we can sleep for a long time
            await asyncio.sleep(3600) # Check every hour

    async def shutdown(self, reason=""):
        logging.info(f"Initiating shutdown of CompliantFundingFeeCollector. Reason: {reason}")
        self.shutdown_event.set()
        if self.monitoring_task:
            self.monitoring_task.cancel()

if __name__ == '__main__':
    print("CompliantFundingFeeCollector structure is defined. Integration testing is required.")
