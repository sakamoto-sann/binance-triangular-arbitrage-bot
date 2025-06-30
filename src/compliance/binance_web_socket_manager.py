
import asyncio
import logging
import websockets
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BinanceWebSocketManager:
    def __init__(self, max_streams_per_connection=5, max_connections=1024):
        self.max_streams_per_connection = max_streams_per_connection
        self.max_connections = max_connections
        self.connections = []
        self.subscriptions = {}
        self.shutdown_event = asyncio.Event()

    async def subscribe(self, stream_name, callback):
        if self.shutdown_event.is_set():
            logging.warning("Shutdown in progress, not accepting new subscriptions.")
            return

        if stream_name in self.subscriptions:
            logging.warning(f"Already subscribed to {stream_name}.")
            return

        self.subscriptions[stream_name] = callback

        for conn in self.connections:
            if len(conn['streams']) < self.max_streams_per_connection:
                await self._add_stream_to_connection(conn, stream_name)
                return

        if len(self.connections) < self.max_connections:
            await self._create_new_connection([stream_name])
        else:
            logging.error("Maximum number of WebSocket connections reached.")
            # In a real scenario, you might want to queue the subscription

    async def unsubscribe(self, stream_name):
        if stream_name not in self.subscriptions:
            logging.warning(f"Not subscribed to {stream_name}.")
            return

        del self.subscriptions[stream_name]

        for conn in self.connections:
            if stream_name in conn['streams']:
                await self._remove_stream_from_connection(conn, stream_name)
                return

    async def _create_new_connection(self, streams):
        base_url = "wss://stream.binance.com:9443/stream?streams="
        url = base_url + "/".join(streams)
        
        try:
            websocket = await websockets.connect(url)
            conn_info = {
                'websocket': websocket,
                'streams': streams,
                'task': asyncio.create_task(self._listen(websocket))
            }
            self.connections.append(conn_info)
            logging.info(f"Created new WebSocket connection for streams: {streams}")
        except Exception as e:
            logging.error(f"Failed to create WebSocket connection: {e}")
            # Implement retry logic here

    async def _listen(self, websocket):
        while not self.shutdown_event.is_set():
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if 'stream' in data and data['stream'] in self.subscriptions:
                    await self.subscriptions[data['stream']](data['data'])
            except websockets.exceptions.ConnectionClosed:
                logging.warning("WebSocket connection closed. Reconnecting...")
                await self._reconnect()
                break # Exit this listener, a new one will be created
            except Exception as e:
                logging.error(f"Error in WebSocket listener: {e}")

    async def _reconnect(self):
        # A simple reconnect strategy. In a real application, this would be more robust.
        await asyncio.sleep(5)
        all_streams = list(self.subscriptions.keys())
        await self.shutdown() # Close existing connections
        self.shutdown_event.clear()
        self.connections.clear()
        
        for i in range(0, len(all_streams), self.max_streams_per_connection):
            chunk = all_streams[i:i+self.max_streams_per_connection]
            await self._create_new_connection(chunk)

    async def shutdown(self):
        logging.info("Initiating shutdown of BinanceWebSocketManager.")
        self.shutdown_event.set()
        for conn in self.connections:
            conn['task'].cancel()
            await conn['websocket'].close()
        self.connections.clear()
        logging.info("BinanceWebSocketManager has been shut down.")

if __name__ == '__main__':
    async def handle_kline(data):
        print(f"Kline data received: {data['k']['c']}")

    async def handle_depth(data):
        print(f"Depth update: {data['b'][0]}")

    async def main():
        ws_manager = BinanceWebSocketManager()
        await ws_manager.subscribe('btcusdt@kline_1m', handle_kline)
        await ws_manager.subscribe('ethusdt@depth', handle_depth)

        await asyncio.sleep(10)
        await ws_manager.shutdown()

    asyncio.run(main())
