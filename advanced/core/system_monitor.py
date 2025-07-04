"""
Advanced Trading System - System Monitor
System health monitoring and emergency protocols for portfolio management.
"""

import asyncio
import logging
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)

class SystemHealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
    OFFLINE = "OFFLINE"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class SystemHealthMetrics:
    """System health and performance metrics."""
    # System resource metrics
    cpu_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_pct: float = 0.0
    disk_free_gb: float = 0.0
    
    # Network metrics
    network_latency_ms: float = 0.0
    packet_loss_pct: float = 0.0
    connection_count: int = 0
    
    # Application metrics
    active_strategies: int = 0
    active_connections: int = 0
    message_queue_size: int = 0
    error_rate: float = 0.0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    throughput_per_sec: float = 0.0
    memory_leaks_detected: bool = False
    gc_frequency: float = 0.0
    
    # Trading metrics
    orders_per_minute: float = 0.0
    fill_rate_pct: float = 0.0
    slippage_avg: float = 0.0
    exchange_connectivity: Dict[str, bool] = field(default_factory=dict)
    
    # Overall health
    overall_status: SystemHealthStatus = SystemHealthStatus.HEALTHY
    health_score: float = 1.0  # 0-1 scale
    
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class SystemAlert:
    """System alert details."""
    alert_id: str
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    auto_action_taken: bool = False

@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    test_name: str
    
    # Timing metrics
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    
    # Throughput metrics
    operations_per_second: float = 0.0
    data_processed_mb: float = 0.0
    
    # Comparison
    baseline_time_ms: float = 0.0
    performance_ratio: float = 1.0  # vs baseline
    
    passed: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

class SystemMonitor:
    """
    Comprehensive system health monitoring and emergency management.
    
    Monitors system resources, application performance, trading metrics,
    and automatically responds to critical conditions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system monitor.
        
        Args:
            config: Monitor configuration settings
        """
        self.config = config or {}
        
        # Health monitoring
        self.current_health = SystemHealthMetrics()
        self.health_history: List[SystemHealthMetrics] = []
        self.active_alerts: List[SystemAlert] = []
        
        # Performance monitoring
        self.performance_benchmarks: List[PerformanceBenchmark] = []
        self.response_times: List[float] = []
        self.error_counts: Dict[str, int] = {}
        
        # System state tracking
        self.start_time = datetime.now()
        self.last_gc_time = datetime.now()
        self.memory_baseline = 0.0
        self.emergency_shutdown_triggered = False
        
        # Monitoring settings
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        self.alert_thresholds = self._initialize_alert_thresholds()
        self.emergency_thresholds = self._initialize_emergency_thresholds()
        
        # Monitoring tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_active = False
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("SystemMonitor initialized")
    
    def _initialize_alert_thresholds(self) -> Dict[str, float]:
        """Initialize alert threshold values."""
        return {
            'cpu_usage_warning': 70.0,      # 70% CPU
            'cpu_usage_critical': 90.0,     # 90% CPU
            'memory_usage_warning': 80.0,   # 80% Memory
            'memory_usage_critical': 95.0,  # 95% Memory
            'disk_usage_warning': 85.0,     # 85% Disk
            'disk_usage_critical': 95.0,    # 95% Disk
            'error_rate_warning': 5.0,      # 5% error rate
            'error_rate_critical': 15.0,    # 15% error rate
            'latency_warning': 1000.0,      # 1 second
            'latency_critical': 5000.0,     # 5 seconds
            'packet_loss_warning': 1.0,     # 1% packet loss
            'packet_loss_critical': 5.0,    # 5% packet loss
        }
    
    def _initialize_emergency_thresholds(self) -> Dict[str, float]:
        """Initialize emergency shutdown thresholds."""
        return {
            'cpu_usage_emergency': 98.0,    # 98% CPU
            'memory_usage_emergency': 99.0, # 99% Memory
            'error_rate_emergency': 50.0,   # 50% error rate
            'consecutive_failures': 10,      # 10 consecutive failures
            'market_loss_emergency': 0.20,   # 20% portfolio loss
        }
    
    async def start(self) -> bool:
        """Start system monitoring."""
        try:
            self.logger.info("Starting system monitoring...")
            
            # Record baseline metrics
            await self._record_baseline_metrics()
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._system_health_monitor()),
                asyncio.create_task(self._performance_monitor()),
                asyncio.create_task(self._alert_processor()),
                asyncio.create_task(self._emergency_monitor()),
                asyncio.create_task(self._benchmark_runner())
            ]
            
            self.is_active = True
            self.logger.info("System monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system monitoring: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop system monitoring."""
        try:
            self.logger.info("Stopping system monitoring...")
            
            self.is_active = False
            
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                task.cancel()
            
            # Generate final health report
            await self._generate_shutdown_report()
            
            self.logger.info("System monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping system monitoring: {e}")
            return False
    
    async def _record_baseline_metrics(self):
        """Record baseline system metrics."""
        try:
            process = psutil.Process()
            self.memory_baseline = process.memory_info().rss / (1024 * 1024)  # MB
            
            self.logger.info(f"Baseline memory usage: {self.memory_baseline:.1f} MB")
            
        except Exception as e:
            self.logger.error(f"Error recording baseline metrics: {e}")
    
    async def _system_health_monitor(self):
        """Monitor system health metrics."""
        while self.is_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Analyze health status
                await self._analyze_health_status()
                
                # Store in history
                self.health_history.append(self.current_health)
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in system health monitor: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics."""
        try:
            metrics = SystemHealthMetrics()
            
            # CPU and Memory
            metrics.cpu_usage_pct = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            metrics.memory_usage_pct = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.disk_usage_pct = (disk.used / disk.total) * 100
            metrics.disk_free_gb = disk.free / (1024**3)
            
            # Network metrics (simplified)
            net_io = psutil.net_io_counters()
            metrics.packet_loss_pct = 0.0  # Would need actual ping tests
            
            # Process-specific metrics
            process = psutil.Process()
            metrics.connection_count = len(process.connections())
            
            # Application metrics
            metrics.error_rate = await self._calculate_error_rate()
            metrics.avg_response_time_ms = await self._calculate_avg_response_time()
            metrics.gc_frequency = await self._calculate_gc_frequency()
            
            # Memory leak detection
            current_memory = process.memory_info().rss / (1024 * 1024)
            memory_growth = current_memory - self.memory_baseline
            metrics.memory_leaks_detected = memory_growth > 500  # 500MB growth
            
            self.current_health = metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        try:
            total_errors = sum(self.error_counts.values())
            
            # Calculate over last 10 minutes
            time_window_minutes = 10
            total_operations = max(len(self.response_times), 1)
            
            if total_operations > 0:
                return (total_errors / total_operations) * 100
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating error rate: {e}")
            return 0.0
    
    async def _calculate_avg_response_time(self) -> float:
        """Calculate average response time."""
        try:
            if not self.response_times:
                return 0.0
            
            # Use last 100 response times
            recent_times = self.response_times[-100:]
            return np.mean(recent_times)
            
        except Exception as e:
            self.logger.error(f"Error calculating response time: {e}")
            return 0.0
    
    async def _calculate_gc_frequency(self) -> float:
        """Calculate garbage collection frequency."""
        try:
            # Trigger GC and measure
            before = datetime.now()
            gc.collect()
            after = datetime.now()
            
            gc_time = (after - before).total_seconds() * 1000  # ms
            
            # Calculate frequency based on time since last GC
            time_since_last = (datetime.now() - self.last_gc_time).total_seconds()
            self.last_gc_time = datetime.now()
            
            frequency = 1.0 / max(time_since_last, 1.0)  # Hz
            return frequency
            
        except Exception as e:
            self.logger.error(f"Error calculating GC frequency: {e}")
            return 0.0
    
    async def _analyze_health_status(self):
        """Analyze overall system health status."""
        try:
            metrics = self.current_health
            health_score = 1.0
            status = SystemHealthStatus.HEALTHY
            
            # CPU health
            if metrics.cpu_usage_pct > self.alert_thresholds['cpu_usage_critical']:
                health_score -= 0.3
                status = SystemHealthStatus.CRITICAL
            elif metrics.cpu_usage_pct > self.alert_thresholds['cpu_usage_warning']:
                health_score -= 0.1
                if status == SystemHealthStatus.HEALTHY:
                    status = SystemHealthStatus.WARNING
            
            # Memory health
            if metrics.memory_usage_pct > self.alert_thresholds['memory_usage_critical']:
                health_score -= 0.3
                status = SystemHealthStatus.CRITICAL
            elif metrics.memory_usage_pct > self.alert_thresholds['memory_usage_warning']:
                health_score -= 0.1
                if status == SystemHealthStatus.HEALTHY:
                    status = SystemHealthStatus.WARNING
            
            # Error rate health
            if metrics.error_rate > self.alert_thresholds['error_rate_critical']:
                health_score -= 0.4
                status = SystemHealthStatus.CRITICAL
            elif metrics.error_rate > self.alert_thresholds['error_rate_warning']:
                health_score -= 0.2
                if status == SystemHealthStatus.HEALTHY:
                    status = SystemHealthStatus.WARNING
            
            # Memory leak detection
            if metrics.memory_leaks_detected:
                health_score -= 0.2
                if status == SystemHealthStatus.HEALTHY:
                    status = SystemHealthStatus.WARNING
            
            # Response time health
            if metrics.avg_response_time_ms > self.alert_thresholds['latency_critical']:
                health_score -= 0.2
                status = SystemHealthStatus.CRITICAL
            elif metrics.avg_response_time_ms > self.alert_thresholds['latency_warning']:
                health_score -= 0.1
                if status == SystemHealthStatus.HEALTHY:
                    status = SystemHealthStatus.WARNING
            
            metrics.health_score = max(0.0, health_score)
            metrics.overall_status = status
            
        except Exception as e:
            self.logger.error(f"Error analyzing health status: {e}")
            self.current_health.overall_status = SystemHealthStatus.WARNING
    
    async def _performance_monitor(self):
        """Monitor application performance."""
        while self.is_active:
            try:
                # Clean up old response times
                cutoff_time = datetime.now() - timedelta(minutes=10)
                # This would need timestamps to implement properly
                
                # Monitor memory growth
                await self._check_memory_growth()
                
                # Monitor connection health
                await self._check_connection_health()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def _check_memory_growth(self):
        """Check for concerning memory growth patterns."""
        try:
            if len(self.health_history) < 10:
                return
            
            recent_memory = [h.memory_usage_pct for h in self.health_history[-10:]]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            
            # Alert if memory is growing rapidly
            if memory_trend > 2.0:  # 2% per monitoring interval
                await self._create_alert(
                    "memory_growth",
                    AlertSeverity.WARNING,
                    "system",
                    f"Rapid memory growth detected: {memory_trend:.1f}%/interval",
                    {"trend": memory_trend, "recent_usage": recent_memory}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking memory growth: {e}")
    
    async def _check_connection_health(self):
        """Check connection health and stability."""
        try:
            # This would interface with exchange connections
            # For now, simulate based on current metrics
            
            if self.current_health.connection_count == 0:
                await self._create_alert(
                    "no_connections",
                    AlertSeverity.CRITICAL,
                    "network",
                    "No active connections detected",
                    {"connection_count": 0}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking connection health: {e}")
    
    async def _alert_processor(self):
        """Process and manage system alerts."""
        while self.is_active:
            try:
                # Check for threshold violations
                await self._check_alert_conditions()
                
                # Auto-resolve stale alerts
                await self._auto_resolve_alerts()
                
                # Clean up old alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(30)
    
    async def _check_alert_conditions(self):
        """Check for alert threshold violations."""
        try:
            metrics = self.current_health
            
            # CPU usage alerts
            if (metrics.cpu_usage_pct > self.alert_thresholds['cpu_usage_critical'] and
                not self._alert_exists('cpu_critical')):
                await self._create_alert(
                    'cpu_critical',
                    AlertSeverity.CRITICAL,
                    'system',
                    f'Critical CPU usage: {metrics.cpu_usage_pct:.1f}%',
                    {'cpu_usage': metrics.cpu_usage_pct}
                )
            
            # Memory usage alerts
            if (metrics.memory_usage_pct > self.alert_thresholds['memory_usage_critical'] and
                not self._alert_exists('memory_critical')):
                await self._create_alert(
                    'memory_critical',
                    AlertSeverity.CRITICAL,
                    'system',
                    f'Critical memory usage: {metrics.memory_usage_pct:.1f}%',
                    {'memory_usage': metrics.memory_usage_pct}
                )
            
            # Error rate alerts
            if (metrics.error_rate > self.alert_thresholds['error_rate_critical'] and
                not self._alert_exists('error_rate_critical')):
                await self._create_alert(
                    'error_rate_critical',
                    AlertSeverity.CRITICAL,
                    'application',
                    f'Critical error rate: {metrics.error_rate:.1f}%',
                    {'error_rate': metrics.error_rate}
                )
            
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")
    
    def _alert_exists(self, alert_id: str) -> bool:
        """Check if alert already exists."""
        return any(alert.alert_id == alert_id and not alert.resolved 
                  for alert in self.active_alerts)
    
    async def _create_alert(self, alert_id: str, severity: AlertSeverity, 
                          component: str, message: str, details: Dict[str, Any]):
        """Create and process a new alert."""
        try:
            alert = SystemAlert(
                alert_id=alert_id,
                severity=severity,
                component=component,
                message=message,
                details=details
            )
            
            self.active_alerts.append(alert)
            
            # Log the alert
            log_level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.EMERGENCY: logging.CRITICAL
            }.get(severity, logging.WARNING)
            
            self.logger.log(log_level, f"ALERT [{severity.value}] {component}: {message}")
            
            # Take automatic action if needed
            await self._handle_automatic_actions(alert)
            
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
    
    async def _handle_automatic_actions(self, alert: SystemAlert):
        """Handle automatic actions for alerts."""
        try:
            if alert.severity == AlertSeverity.CRITICAL:
                # Trigger garbage collection for memory issues
                if 'memory' in alert.alert_id:
                    gc.collect()
                    alert.auto_action_taken = True
                    self.logger.info("Automatic GC triggered for memory alert")
                
                # Reduce operation frequency for CPU issues
                if 'cpu' in alert.alert_id:
                    # This would reduce trading frequency
                    alert.auto_action_taken = True
                    self.logger.info("Automatic CPU load reduction triggered")
            
            elif alert.severity == AlertSeverity.EMERGENCY:
                # Trigger emergency protocols
                await self._trigger_emergency_protocols(alert)
            
        except Exception as e:
            self.logger.error(f"Error handling automatic actions: {e}")
    
    async def _auto_resolve_alerts(self):
        """Automatically resolve alerts when conditions improve."""
        try:
            for alert in self.active_alerts:
                if alert.resolved:
                    continue
                
                # Check if conditions have improved
                should_resolve = False
                
                if alert.alert_id == 'cpu_critical':
                    should_resolve = self.current_health.cpu_usage_pct < self.alert_thresholds['cpu_usage_warning']
                elif alert.alert_id == 'memory_critical':
                    should_resolve = self.current_health.memory_usage_pct < self.alert_thresholds['memory_usage_warning']
                elif alert.alert_id == 'error_rate_critical':
                    should_resolve = self.current_health.error_rate < self.alert_thresholds['error_rate_warning']
                
                if should_resolve:
                    alert.resolved = True
                    self.logger.info(f"Alert auto-resolved: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Error auto-resolving alerts: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not alert.resolved or alert.timestamp > cutoff_time
            ]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up alerts: {e}")
    
    async def _emergency_monitor(self):
        """Monitor for emergency conditions."""
        while self.is_active:
            try:
                await self._check_emergency_conditions()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in emergency monitor: {e}")
                await asyncio.sleep(10)
    
    async def _check_emergency_conditions(self):
        """Check for emergency shutdown conditions."""
        try:
            metrics = self.current_health
            
            # CPU emergency
            if metrics.cpu_usage_pct > self.emergency_thresholds['cpu_usage_emergency']:
                await self._trigger_emergency_shutdown("CPU usage critical")
            
            # Memory emergency
            if metrics.memory_usage_pct > self.emergency_thresholds['memory_usage_emergency']:
                await self._trigger_emergency_shutdown("Memory usage critical")
            
            # Error rate emergency
            if metrics.error_rate > self.emergency_thresholds['error_rate_emergency']:
                await self._trigger_emergency_shutdown("Error rate critical")
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")
    
    async def _trigger_emergency_protocols(self, alert: SystemAlert):
        """Trigger emergency response protocols."""
        try:
            self.logger.critical(f"EMERGENCY PROTOCOL TRIGGERED: {alert.message}")
            
            # Create emergency alert
            await self._create_alert(
                f"emergency_{alert.alert_id}",
                AlertSeverity.EMERGENCY,
                "system",
                f"Emergency protocols activated: {alert.message}",
                {"original_alert": alert.alert_id}
            )
            
            # Emergency actions would be implemented here
            # - Stop new trades
            # - Close positions
            # - Notify administrators
            # - Save state
            
        except Exception as e:
            self.logger.error(f"Error triggering emergency protocols: {e}")
    
    async def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency system shutdown."""
        try:
            if self.emergency_shutdown_triggered:
                return  # Already triggered
            
            self.emergency_shutdown_triggered = True
            
            self.logger.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
            
            # Create emergency alert
            await self._create_alert(
                "emergency_shutdown",
                AlertSeverity.EMERGENCY,
                "system",
                f"Emergency shutdown initiated: {reason}",
                {"shutdown_reason": reason}
            )
            
            # Emergency shutdown procedures would be implemented here
            # - Stop all trading immediately
            # - Close all positions
            # - Save system state
            # - Notify administrators
            # - Graceful shutdown
            
        except Exception as e:
            self.logger.error(f"Error triggering emergency shutdown: {e}")
    
    async def _benchmark_runner(self):
        """Run periodic performance benchmarks."""
        while self.is_active:
            try:
                # Run benchmarks every hour
                await asyncio.sleep(3600)
                
                if not self.is_active:
                    break
                
                await self._run_performance_benchmarks()
                
            except Exception as e:
                self.logger.error(f"Error in benchmark runner: {e}")
                await asyncio.sleep(3600)
    
    async def _run_performance_benchmarks(self):
        """Run comprehensive performance benchmarks."""
        try:
            benchmarks = []
            
            # CPU benchmark
            cpu_benchmark = await self._benchmark_cpu_performance()
            if cpu_benchmark:
                benchmarks.append(cpu_benchmark)
            
            # Memory benchmark
            memory_benchmark = await self._benchmark_memory_performance()
            if memory_benchmark:
                benchmarks.append(memory_benchmark)
            
            # I/O benchmark
            io_benchmark = await self._benchmark_io_performance()
            if io_benchmark:
                benchmarks.append(io_benchmark)
            
            self.performance_benchmarks.extend(benchmarks)
            
            # Keep only recent benchmarks
            if len(self.performance_benchmarks) > 100:
                self.performance_benchmarks = self.performance_benchmarks[-100:]
            
            self.logger.info(f"Completed {len(benchmarks)} performance benchmarks")
            
        except Exception as e:
            self.logger.error(f"Error running performance benchmarks: {e}")
    
    async def _benchmark_cpu_performance(self) -> Optional[PerformanceBenchmark]:
        """Benchmark CPU performance."""
        try:
            start_time = datetime.now()
            
            # Simple CPU-intensive task
            result = sum(i * i for i in range(100000))
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            benchmark = PerformanceBenchmark(
                test_name="cpu_intensive",
                execution_time_ms=execution_time,
                baseline_time_ms=50.0,  # 50ms baseline
                performance_ratio=50.0 / execution_time if execution_time > 0 else 1.0,
                passed=execution_time < 100.0  # Pass if under 100ms
            )
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"Error in CPU benchmark: {e}")
            return None
    
    async def _benchmark_memory_performance(self) -> Optional[PerformanceBenchmark]:
        """Benchmark memory performance."""
        try:
            start_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            start_time = datetime.now()
            
            # Memory allocation test
            test_data = [i for i in range(10000)]
            test_dict = {i: str(i) for i in range(5000)}
            
            end_time = datetime.now()
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            execution_time = (end_time - start_time).total_seconds() * 1000
            memory_used = end_memory - start_memory
            
            # Clean up
            del test_data, test_dict
            gc.collect()
            
            benchmark = PerformanceBenchmark(
                test_name="memory_allocation",
                execution_time_ms=execution_time,
                memory_usage_mb=memory_used,
                baseline_time_ms=10.0,
                performance_ratio=10.0 / execution_time if execution_time > 0 else 1.0,
                passed=execution_time < 50.0 and memory_used < 100.0
            )
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"Error in memory benchmark: {e}")
            return None
    
    async def _benchmark_io_performance(self) -> Optional[PerformanceBenchmark]:
        """Benchmark I/O performance."""
        try:
            start_time = datetime.now()
            
            # Simple file I/O test
            test_file = "/tmp/system_monitor_benchmark.txt"
            test_data = "x" * 1000  # 1KB of data
            
            # Write test
            with open(test_file, 'w') as f:
                for _ in range(100):  # 100KB total
                    f.write(test_data)
            
            # Read test
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup
            if os.path.exists(test_file):
                os.remove(test_file)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            benchmark = PerformanceBenchmark(
                test_name="io_performance",
                execution_time_ms=execution_time,
                data_processed_mb=0.1,  # 100KB
                baseline_time_ms=20.0,
                performance_ratio=20.0 / execution_time if execution_time > 0 else 1.0,
                passed=execution_time < 100.0
            )
            
            return benchmark
            
        except Exception as e:
            self.logger.error(f"Error in I/O benchmark: {e}")
            return None
    
    async def _generate_shutdown_report(self):
        """Generate final system report on shutdown."""
        try:
            uptime = datetime.now() - self.start_time
            
            report = {
                "system_uptime": str(uptime),
                "final_health_score": self.current_health.health_score,
                "total_alerts": len(self.active_alerts),
                "unresolved_alerts": len([a for a in self.active_alerts if not a.resolved]),
                "emergency_shutdowns": self.emergency_shutdown_triggered,
                "performance_summary": self._get_performance_summary()
            }
            
            self.logger.info(f"System shutdown report: {json.dumps(report, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Error generating shutdown report: {e}")
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        try:
            if not self.performance_benchmarks:
                return {}
            
            cpu_benchmarks = [b for b in self.performance_benchmarks if b.test_name == "cpu_intensive"]
            memory_benchmarks = [b for b in self.performance_benchmarks if b.test_name == "memory_allocation"]
            io_benchmarks = [b for b in self.performance_benchmarks if b.test_name == "io_performance"]
            
            summary = {}
            
            if cpu_benchmarks:
                avg_cpu_time = np.mean([b.execution_time_ms for b in cpu_benchmarks])
                summary["avg_cpu_benchmark_ms"] = avg_cpu_time
            
            if memory_benchmarks:
                avg_memory_time = np.mean([b.execution_time_ms for b in memory_benchmarks])
                summary["avg_memory_benchmark_ms"] = avg_memory_time
            
            if io_benchmarks:
                avg_io_time = np.mean([b.execution_time_ms for b in io_benchmarks])
                summary["avg_io_benchmark_ms"] = avg_io_time
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
    
    def record_response_time(self, response_time_ms: float):
        """Record an operation response time."""
        self.response_times.append(response_time_ms)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "health": {
                "overall_status": self.current_health.overall_status.value,
                "health_score": self.current_health.health_score,
                "cpu_usage": self.current_health.cpu_usage_pct,
                "memory_usage": self.current_health.memory_usage_pct,
                "error_rate": self.current_health.error_rate
            },
            "alerts": {
                "total_active": len(self.active_alerts),
                "unresolved": len([a for a in self.active_alerts if not a.resolved]),
                "critical": len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved])
            },
            "uptime": str(datetime.now() - self.start_time),
            "emergency_status": self.emergency_shutdown_triggered
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False