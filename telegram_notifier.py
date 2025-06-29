#!/usr/bin/env python3
"""
Telegram Notifications for Triangular Arbitrage Bot
Handles all Telegram notifications with rate limiting and error handling
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Optional

# Telegram imports with fallback for missing library
try:
    import telegram
    from telegram.constants import ParseMode
    from telegram.error import TelegramError, NetworkError, TimedOut
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    telegram = None
    ParseMode = None
    TelegramError = Exception
    NetworkError = Exception
    TimedOut = Exception

import config

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Handles all Telegram notifications for the trading bot"""
    
    def __init__(self):
        self.enabled = config.TELEGRAM_ENABLED and TELEGRAM_AVAILABLE
        
        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("Telegram library not available. Install python-telegram-bot to enable notifications")
            elif not config.TELEGRAM_BOT_TOKEN:
                logger.warning("TELEGRAM_BOT_TOKEN not set. Telegram notifications disabled")
            elif not config.TELEGRAM_CHAT_IDS:
                logger.warning("TELEGRAM_CHAT_IDS not set. Telegram notifications disabled")
            return
        
        try:
            self.bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
            self.chat_ids = config.TELEGRAM_CHAT_IDS
            self.settings = config.NOTIFICATION_SETTINGS
            
            # Rate limiting tracking
            self.message_history = defaultdict(deque)
            self.last_sent = defaultdict(float)
            
            # Daily summary tracking
            self.daily_pnl_summary_sent = False
            self.daily_reset_time = time.time()
            
            logger.info(f"Telegram notifier initialized for {len(self.chat_ids)} chat(s)")
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enabled = False
    
    async def send_trade_execution(self, trade_result: Dict):
        """Send trade execution notification"""
        if not self.enabled or not self._should_send_notification("trade_execution"):
            return
        
        try:
            if trade_result["success"]:
                profit = trade_result["profit"]
                profit_pct = trade_result["profit_percentage"]
                initial_amount = trade_result["initial_amount"]
                final_amount = trade_result["final_amount"]
                
                # Build symbol chain from trades
                symbol_chain = []
                for trade in trade_result.get("trades", []):
                    symbol_chain.append(trade.get("symbol", ""))
                symbol_display = " → ".join(filter(None, symbol_chain))
                
                profit_emoji = "🟢" if profit > 0 else "🔴"
                
                message = f"""
{profit_emoji} *Trade Executed Successfully*

💱 *Symbols*: {symbol_display}
💰 *Initial*: {initial_amount:.6f}
💰 *Final*: {final_amount:.6f}
📈 *Profit*: {profit:.6f} ({profit_pct:.4f}%)
⏰ *Time*: {datetime.now().strftime('%H:%M:%S')}
                """.strip()
            else:
                error_msg = trade_result.get("error", "Unknown error")
                message = f"""
🔴 *Trade Failed*

❌ *Error*: {error_msg}
⏰ *Time*: {datetime.now().strftime('%H:%M:%S')}
                """.strip()
            
            await self._send_notification(message, "trade_execution")
            
        except Exception as e:
            logger.error(f"Error sending trade execution notification: {e}")
    
    async def send_opportunity_alert(self, triangle: List[str], profit_percentage: float):
        """Send arbitrage opportunity alert"""
        if not self.enabled or not self._should_send_notification("opportunity"):
            return
        
        try:
            triangle_display = " → ".join(triangle)
            profit_pct_display = profit_percentage * 100
            
            message = f"""
💡 *Arbitrage Opportunity*

🔄 *Triangle*: {triangle_display}
📈 *Potential Profit*: {profit_pct_display:.4f}%
⏰ *Time*: {datetime.now().strftime('%H:%M:%S')}
            """.strip()
            
            await self._send_notification(message, "opportunity")
            
        except Exception as e:
            logger.error(f"Error sending opportunity alert: {e}")
    
    async def send_error_alert(self, error_type: str, error_message: str, severity: str = "medium"):
        """Send error notifications"""
        if not self.enabled or not self._should_send_notification("error"):
            return
        
        try:
            emoji_map = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}
            emoji = emoji_map.get(severity, "⚠️")
            
            # Sanitize error message to prevent sensitive info leakage
            safe_error_msg = self._sanitize_error_message(error_message)
            
            message = f"""
{emoji} *{error_type} Error*

⚠️ *Message*: {safe_error_msg}
🔍 *Severity*: {severity.upper()}
⏰ *Time*: {datetime.now().strftime('%H:%M:%S')}
            """.strip()
            
            await self._send_notification(message, "error")
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
    
    async def send_daily_summary(self, pnl_data: Dict):
        """Send daily P&L summary"""
        if not self.enabled or self.daily_pnl_summary_sent:
            return
        
        try:
            total_pnl = pnl_data.get("total_pnl", 0)
            trades_count = pnl_data.get("trades_count", 0)
            successful_trades = pnl_data.get("successful_trades", 0)
            failed_trades = pnl_data.get("failed_trades", 0)
            
            success_rate = (successful_trades / trades_count * 100) if trades_count > 0 else 0
            
            pnl_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            
            message = f"""
📊 *Daily Trading Summary*

{pnl_emoji} *Total P&L*: {total_pnl:.6f} USDT
📈 *Trades*: {trades_count} total
✅ *Successful*: {successful_trades}
❌ *Failed*: {failed_trades}
🎯 *Success Rate*: {success_rate:.1f}%
📅 *Date*: {datetime.now().strftime('%Y-%m-%d')}
            """.strip()
            
            await self._send_notification(message, "daily_summary")
            self.daily_pnl_summary_sent = True
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    async def send_bot_status(self, status: str, additional_info: str = ""):
        """Send bot status updates"""
        if not self.enabled or not self._should_send_notification("status"):
            return
        
        try:
            status_emojis = {
                "starting": "🟡",
                "running": "🟢", 
                "stopping": "🟠",
                "stopped": "🔴",
                "error": "❌",
                "reconnecting": "🔄",
                "connected": "✅",
                "disconnected": "⚠️"
            }
            
            emoji = status_emojis.get(status.lower(), "ℹ️")
            
            message = f"""
{emoji} *Bot Status: {status.upper()}*

⏰ *Time*: {datetime.now().strftime('%H:%M:%S')}
            """
            
            if additional_info:
                # Sanitize additional info
                safe_info = self._sanitize_error_message(additional_info)
                message += f"\nℹ️ *Info*: {safe_info}"
            
            message = message.strip()
            await self._send_notification(message, "status")
            
        except Exception as e:
            logger.error(f"Error sending bot status: {e}")
    
    def _should_send_notification(self, notification_type: str) -> bool:
        """Check if notification should be sent based on rate limits"""
        if not self.enabled:
            return False
        
        settings = self.settings.get(notification_type, {})
        if not settings.get("enabled", False):
            return False
        
        current_time = time.time()
        
        # Check minimum time between messages
        rate_limit = settings.get("rate_limit_seconds", 0)
        if current_time - self.last_sent[notification_type] < rate_limit:
            return False
        
        # Check hourly rate limit
        history = self.message_history[notification_type]
        hour_ago = current_time - 3600
        
        # Clean old entries
        while history and history[0] < hour_ago:
            history.popleft()
        
        max_per_hour = settings.get("max_per_hour", 60)
        if len(history) >= max_per_hour:
            return False
        
        return True
    
    async def _send_notification(self, message: str, notification_type: str):
        """Send notification to all configured chat IDs"""
        if not self.enabled:
            return
        
        current_time = time.time()
        
        # Reset daily summary flag if new day
        if current_time - self.daily_reset_time > 86400:
            self.daily_pnl_summary_sent = False
            self.daily_reset_time = current_time
        
        for chat_id in self.chat_ids:
            try:
                # Sanitize ALL messages before sending (extra security layer)
                safe_message = self._sanitize_error_message(message)
                
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=safe_message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
                
                # Update rate limiting trackers
                self.last_sent[notification_type] = current_time
                self.message_history[notification_type].append(current_time)
                
                # Small delay between messages to avoid rate limits
                await asyncio.sleep(0.1)
                
            except (NetworkError, TimedOut) as e:
                logger.warning(f"Network error sending Telegram message to {chat_id}: {e}")
                # Retry once after delay
                await asyncio.sleep(1)
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=message,
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=True
                    )
                except Exception as retry_e:
                    logger.error(f"Failed to send Telegram message to {chat_id} after retry: {retry_e}")
                    
            except TelegramError as e:
                logger.error(f"Telegram API error sending to {chat_id}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error sending Telegram message to {chat_id}: {e}")
    
    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error messages to prevent sensitive information leakage"""
        if not message:
            return "Unknown error"
        
        # Remove potential API keys, secrets, and sensitive patterns
        import re
        
        # Enhanced patterns to sanitize - covers more cases
        patterns = [
            # API Keys and Secrets
            (r'[A-Za-z0-9]{64}', '[API_KEY_HIDDEN]'),  # 64-char API keys
            (r'[A-Za-z0-9+/]{40,}={0,2}', '[SECRET_HIDDEN]'),  # Base64-like secrets
            (r'sk-[A-Za-z0-9]{20,}', '[SECRET_KEY_HIDDEN]'),  # OpenAI-style keys
            
            # Various authentication patterns
            (r'(?i)(token|key|secret|password|auth)[=:\s]*[A-Za-z0-9_+-]{8,}', r'\1=[HIDDEN]'),
            (r'(?i)(bearer|basic)\s+[A-Za-z0-9+/=_-]{8,}', r'\1 [HIDDEN]'),
            
            # Binance-specific patterns
            (r'(?i)(binance[_-]?api[_-]?key)[=:\s]*[A-Za-z0-9]{10,}', r'\1=[HIDDEN]'),
            (r'(?i)(binance[_-]?secret)[=:\s]*[A-Za-z0-9]{10,}', r'\1=[HIDDEN]'),
            
            # General sensitive data patterns
            (r'(?i)(client[_-]?id)[=:\s]*[A-Za-z0-9_-]{8,}', r'\1=[HIDDEN]'),
            (r'(?i)(access[_-]?token)[=:\s]*[A-Za-z0-9_.-]{8,}', r'\1=[HIDDEN]'),
            
            # Remove full HTTP authorization headers
            (r'(?i)authorization:\s*[^\s,]+', 'authorization: [HIDDEN]'),
            
            # Remove JSON with sensitive fields
            (r'(?i)"(api_?key|secret|token|password)":\s*"[^"]{8,}"', r'"\1": "[HIDDEN]"'),
            
            # Remove query parameters with sensitive data
            (r'(?i)[?&](key|token|secret|auth)[=][A-Za-z0-9_+-]{8,}', r'&\1=[HIDDEN]'),
        ]
        
        sanitized = str(message)
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        # Remove any remaining long alphanumeric strings that might be sensitive
        sanitized = re.sub(r'\b[A-Za-z0-9+/]{32,}\b', '[SENSITIVE_DATA_HIDDEN]', sanitized)
        
        # Truncate very long messages
        if len(sanitized) > 300:
            sanitized = sanitized[:297] + "..."
        
        return sanitized
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        if not self.enabled:
            return False
        
        try:
            me = await self.bot.get_me()
            logger.info(f"Telegram bot connected: @{me.username}")
            
            # Send test message to all chat IDs
            test_message = f"""
🧪 *Test Message*

✅ Bot connection successful
🤖 *Bot*: @{me.username}
⏰ *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()
            
            for chat_id in self.chat_ids:
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=test_message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    logger.info(f"Test message sent to chat {chat_id}")
                except Exception as e:
                    logger.error(f"Failed to send test message to chat {chat_id}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False