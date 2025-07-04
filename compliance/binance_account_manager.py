
import logging
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceAccountManager:
    def __init__(self, binance_client):
        self.client = binance_client
        self.account_info = None
        self.shutdown_event = asyncio.Event()

    async def initialize(self):
        if not await self.validate_account_status():
            await self.shutdown("Failed to initialize account. Check permissions and status.")
            return False
        logging.info("Account validation successful.")
        return True

    async def validate_account_status(self):
        if self.shutdown_event.is_set():
            return False
        try:
            self.account_info = await self.client.get_account()
            if not self.account_info['canTrade']:
                logging.error("Account does not have trading permissions.")
                return False
            # Add more status checks as needed (e.g., canWithdraw, canDeposit)
            return True
        except Exception as e:
            logging.error(f"Failed to get account information: {e}")
            return False

    async def validate_trading_permissions(self, symbol):
        if not self.account_info:
            if not await self.validate_account_status():
                return False

        permissions = self.account_info.get('permissions', [])
        if "SPOT" not in permissions and "MARGIN" not in permissions:
             logging.warning(f"Account may not have spot or margin trading permissions.")
        # This is a simplified check. A real implementation would need to check the symbol against the permissions.
        return True

    async def validate_balance_requirements(self, asset, required_amount):
        if self.shutdown_event.is_set():
            return False
        try:
            balance = await self.client.get_asset_balance(asset=asset)
            if not balance or float(balance['free']) < required_amount:
                logging.error(f"Insufficient balance for {asset}. Required: {required_amount}, Available: {balance['free'] if balance else 0}")
                return False
            return True
        except Exception as e:
            logging.error(f"Failed to get balance for {asset}: {e}")
            return False

    async def shutdown(self, reason=""):
        logging.info(f"Initiating shutdown of BinanceAccountManager. Reason: {reason}")
        self.shutdown_event.set()
        # Trigger a global shutdown of the bot
        logging.critical("BinanceAccountManager has triggered an emergency shutdown.")

if __name__ == '__main__':
    # This class requires a mock binance_client to run, which is complex to set up here.
    # The logic is straightforward and will be tested during integration with the main bot.
    print("BinanceAccountManager structure is defined. Integration testing is required.")
