
import asyncio
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceAPIRateLimiter:
    def __init__(self, safety_margin=0.7):
        self.spot_weight_limits = deque()
        self.futures_weight_limits = deque()
        self.order_limits = deque()
        self.safety_margin = safety_margin
        self.shutdown_event = asyncio.Event()

    async def wait_for_request_weight(self, weight: int, is_spot: bool = True):
        if self.shutdown_event.is_set():
            logging.warning("Shutdown in progress, not accepting new requests.")
            return False
        
        limit_queue = self.spot_weight_limits if is_spot else self.futures_weight_limits
        limit_type = "spot" if is_spot else "futures"
        
        # Clean up old requests
        current_time = time.time()
        while limit_queue and limit_queue[0] < current_time - 60:
            limit_queue.popleft()

        # Check if we have capacity
        # Spot: 1200 weight per minute, Futures: 2400 weight per minute
        limit = 1200 * self.safety_margin if is_spot else 2400 * self.safety_margin
        if len(limit_queue) + weight > limit:
            logging.warning(f"Approaching {limit_type} request weight limit. Waiting...")
            await asyncio.sleep(5) # Wait and re-evaluate
            return await self.wait_for_request_weight(weight, is_spot)

        for _ in range(weight):
            limit_queue.append(time.time())
        return True

    async def wait_for_order(self):
        if self.shutdown_event.is_set():
            logging.warning("Shutdown in progress, not accepting new orders.")
            return False

        # Clean up old orders
        current_time = time.time()
        while self.order_limits and self.order_limits[0] < current_time - 10: # 10 seconds for order limits
            self.order_limits.popleft()

        if len(self.order_limits) >= 40: # 70% of 50 orders per 10 seconds (NOT 100)
            logging.warning("Approaching order limit. Waiting...")
            await asyncio.sleep(1) # Wait and re-evaluate
            return await self.wait_for_order()

        self.order_limits.append(time.time())
        return True

    def get_current_usage(self):
        current_time = time.time()
        
        spot_weight = len([t for t in self.spot_weight_limits if t >= current_time - 60])
        futures_weight = len([t for t in self.futures_weight_limits if t >= current_time - 60])
        orders = len([t for t in self.order_limits if t >= current_time - 10])

        return {
            "spot_weight_usage": spot_weight,
            "futures_weight_usage": futures_weight,
            "order_usage": orders
        }

    async def shutdown(self):
        logging.info("Initiating shutdown of BinanceAPIRateLimiter.")
        self.shutdown_event.set()
        # Potentially add more logic here to wait for pending tasks to complete if necessary
        logging.info("BinanceAPIRateLimiter has been shut down.")

if __name__ == '__main__':
    async def main():
        rate_limiter = BinanceAPIRateLimiter()

        async def make_requests():
            for i in range(10):
                if await rate_limiter.wait_for_request_weight(10):
                    logging.info(f"Request {i+1} made.")
                    print(rate_limiter.get_current_usage())
                else:
                    logging.error("Failed to make request.")
                await asyncio.sleep(0.1)

        async def place_orders():
            for i in range(7):
                if await rate_limiter.wait_for_order():
                    logging.info(f"Order {i+1} placed.")
                    print(rate_limiter.get_current_usage())
                else:
                    logging.error("Failed to place order.")
                await asyncio.sleep(0.1)
        
        await asyncio.gather(make_requests(), place_orders())
        await rate_limiter.shutdown()

    asyncio.run(main())
