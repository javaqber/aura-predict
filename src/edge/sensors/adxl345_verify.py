"""
AuraPredict — ADXL345 Hardware Verification  (Fase 8)
======================================================
Standalone verification tool for the ADXL345 sensor.

Run this BEFORE starting the edge scheduler on a new Raspberry Pi
installation to confirm the sensor is wired correctly and responding.

Usage (on Raspberry Pi):
    python -m edge.sensors.adxl345_verify

    # With custom bus/address:
    python -m edge.sensors.adxl345_verify --bus 1 --address 0x53

    # Silent mode (only exit code):
    python -m edge.sensors.adxl345_verify --quiet

    # JSON output (for scripting):
    python -m edge.sensors.adxl345_verify --json

Exit codes:
    0 — All checks passed
    1 — One or more checks failed
    2 — I2C bus unavailable (smbus2 not installed, or no /dev/i2c-1)

Checks performed:
    1. smbus2 is importable
    2. I2C bus can be opened
    3. DEVID register (0x00) == 0xE5  ← most common failure point
    4. DATA_FORMAT register write/readback
    5. BW_RATE register write/readback
    6. Measurement mode can be enabled
    7. 10 consecutive samples are readable
    8. Samples are not constant (sensor alive)
    9. Samples within expected physical range (±2g + 20% headroom)
   10. Back to standby mode on exit

Injectable bus parameter for testability — no hardware needed for tests.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─── CONSTANTS (duplicated to keep this module self-contained) ────────────────

REG_DEVID       = 0x00
REG_BW_RATE     = 0x2C
REG_POWER_CTL   = 0x2D
REG_DATA_FORMAT = 0x31
REG_DATAX0      = 0x32

DEVID_EXPECTED  = 0xE5
DEFAULT_ADDRESS = 0x53
DEFAULT_BUS     = 1


# ─── RESULT ───────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Result of the ADXL345 hardware verification sequence."""
    passed:       bool
    checks:       dict[str, bool]   = field(default_factory=dict)
    errors:       list[str]         = field(default_factory=list)
    warnings:     list[str]         = field(default_factory=list)
    details:      dict              = field(default_factory=dict)

    @property
    def failed_checks(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]

    def summary(self) -> str:
        passed_n = sum(1 for v in self.checks.values() if v)
        total_n  = len(self.checks)
        status   = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} — {passed_n}/{total_n} checks passed\n"
            + ("\n".join(f"  ❌ {c}" for c in self.failed_checks) if not self.passed else "")
            + ("\n" + "\n".join(f"  ⚠️  {w}" for w in self.warnings) if self.warnings else "")
        )


# ─── VERIFICATION FUNCTION ────────────────────────────────────────────────────

def verify_adxl345(
    bus_number:  int           = DEFAULT_BUS,
    address:     int           = DEFAULT_ADDRESS,
    n_test_samples: int        = 10,
    range_g:     int           = 2,
    bus:         Optional[Any] = None,   # Injectable for tests
) -> VerificationResult:
    """
    Run a comprehensive hardware verification sequence on the ADXL345.

    Args:
        bus_number:  I2C bus number (1 on most Raspberry Pi models).
        address:     I2C address (0x53 or 0x1D depending on ALT ADDRESS pin).
        n_test_samples: Number of sample reads to attempt (default 10).
        range_g:     Expected range in g (2, 4, 8, or 16).
        bus:         Injectable I2C bus for tests. None = use real smbus2.

    Returns:
        VerificationResult with pass/fail per check.
    """
    checks:   dict[str, bool] = {}
    errors:   list[str]       = []
    warnings: list[str]       = []
    details:  dict            = {"bus_number": bus_number, "address": hex(address)}

    # ── Check 1: smbus2 importable ─────────────────────────────────────────────
    if bus is not None:
        checks["1_smbus2_importable"] = True
        own_bus = False
    else:
        try:
            import smbus2 as _smbus2
            checks["1_smbus2_importable"] = True
        except ImportError:
            checks["1_smbus2_importable"] = False
            errors.append(
                "smbus2 not installed. Run: pip install smbus2\n"
                "  This is required on Raspberry Pi but not on Windows."
            )
            return VerificationResult(passed=False, checks=checks, errors=errors)

        # ── Check 2: I2C bus openable ──────────────────────────────────────────
        try:
            bus = _smbus2.SMBus(bus_number)
            own_bus = True
            checks["2_i2c_bus_open"] = True
            details["bus_opened"] = f"/dev/i2c-{bus_number}"
        except Exception as exc:
            checks["2_i2c_bus_open"] = False
            errors.append(
                f"Cannot open I2C bus {bus_number}: {exc}\n"
                "  Check: sudo raspi-config → Interface Options → I2C → Enable\n"
                "  Check: sudo chmod a+rw /dev/i2c-1"
            )
            return VerificationResult(passed=False, checks=checks, errors=errors)

    try:
        # ── Check 3: DEVID register ────────────────────────────────────────────
        try:
            devid = bus.read_i2c_block_data(address, REG_DEVID, 1)[0]
            details["devid_read"] = hex(devid)
            checks["3_devid_correct"] = (devid == DEVID_EXPECTED)
            if not checks["3_devid_correct"]:
                errors.append(
                    f"DEVID mismatch at address 0x{address:02X}: "
                    f"expected 0x{DEVID_EXPECTED:02X}, got 0x{devid:02X}.\n"
                    "  Possible causes:\n"
                    "  • Wrong I2C address — try 0x1D if ALT ADDRESS pin is LOW\n"
                    "  • Wiring error (SDA/SCL swapped, missing pull-ups)\n"
                    "  • Sensor not powered\n"
                    "  • Another device at this address"
                )
        except Exception as exc:
            checks["3_devid_correct"] = False
            errors.append(
                f"Cannot read DEVID from 0x{address:02X}: {exc}\n"
                "  The device does not respond. Check:\n"
                "  • VCC connected to 3.3V (NOT 5V)\n"
                "  • GND connected\n"
                "  • SDA → GPIO2 (Pin 3), SCL → GPIO3 (Pin 5)\n"
                "  • i2cdetect -y 1 shows address 0x53"
            )

        if not checks.get("3_devid_correct"):
            return VerificationResult(passed=False, checks=checks, errors=errors,
                                       warnings=warnings, details=details)

        # ── Check 4: DATA_FORMAT write/readback ────────────────────────────────
        _range_map = {2: 0x00, 4: 0x01, 8: 0x02, 16: 0x03}
        range_reg  = _range_map.get(range_g, 0x00)
        data_format = range_reg | 0x08   # FULL_RES bit
        try:
            bus.write_byte_data(address, REG_DATA_FORMAT, data_format)
            readback = bus.read_i2c_block_data(address, REG_DATA_FORMAT, 1)[0]
            checks["4_data_format_rw"] = (readback == data_format)
            details["data_format_written"]  = hex(data_format)
            details["data_format_readback"] = hex(readback)
            if not checks["4_data_format_rw"]:
                warnings.append(
                    f"DATA_FORMAT readback mismatch: wrote 0x{data_format:02X}, "
                    f"read 0x{readback:02X}. Possible I2C noise."
                )
        except Exception as exc:
            checks["4_data_format_rw"] = False
            errors.append(f"DATA_FORMAT register write failed: {exc}")

        # ── Check 5: BW_RATE write/readback ───────────────────────────────────
        # Use 400 Hz (0x0C) for verification — safe on all hardware
        bw_rate = 0x0C
        try:
            bus.write_byte_data(address, REG_BW_RATE, bw_rate)
            readback = bus.read_i2c_block_data(address, REG_BW_RATE, 1)[0]
            checks["5_bw_rate_rw"] = ((readback & 0x0F) == bw_rate)
            details["bw_rate_written"]  = hex(bw_rate)
            details["bw_rate_readback"] = hex(readback)
        except Exception as exc:
            checks["5_bw_rate_rw"] = False
            errors.append(f"BW_RATE register write failed: {exc}")

        # ── Check 6: Enable measurement mode ──────────────────────────────────
        try:
            bus.write_byte_data(address, REG_POWER_CTL, 0x08)
            readback = bus.read_i2c_block_data(address, REG_POWER_CTL, 1)[0]
            checks["6_measurement_mode"] = bool(readback & 0x08)
            if not checks["6_measurement_mode"]:
                warnings.append(
                    f"POWER_CTL readback 0x{readback:02X} — Measure bit not set."
                )
        except Exception as exc:
            checks["6_measurement_mode"] = False
            errors.append(f"POWER_CTL write failed: {exc}")

        if not checks.get("6_measurement_mode"):
            return VerificationResult(passed=False, checks=checks, errors=errors,
                                       warnings=warnings, details=details)

        # Allow sensor to enter measurement mode
        time.sleep(0.05)

        # ── Check 7: Read N samples ────────────────────────────────────────────
        import numpy as np
        scale_map = {0x00: 0.0039, 0x01: 0.0078, 0x02: 0.0156, 0x03: 0.0313}
        scale = scale_map.get(range_reg, 0.0039)
        x_samples = []
        read_errors = 0

        for _ in range(n_test_samples):
            try:
                raw = bus.read_i2c_block_data(address, REG_DATAX0, 6)
                val = ((raw[1] << 8) | raw[0])
                if val >= 0x8000: val -= 0x10000
                x_samples.append(val * scale)
            except Exception as exc:
                read_errors += 1
                logger.debug("Sample read error: %s", exc)

        checks["7_samples_readable"] = (read_errors == 0)
        details["samples_read"]   = n_test_samples - read_errors
        details["read_errors"]    = read_errors
        if read_errors > 0:
            errors.append(
                f"{read_errors}/{n_test_samples} sample reads failed. "
                "Possible: I2C cable too long, missing 4.7kΩ pull-ups, "
                "or electromagnetic interference."
            )

        # ── Check 8: Samples not constant (sensor alive) ──────────────────────
        if x_samples:
            arr = np.array(x_samples)
            std = float(np.std(arr))
            checks["8_non_constant"] = (std > 1e-4)  # at least 0.1 mg variation
            details["x_mean_g"] = round(float(np.mean(arr)), 4)
            details["x_std_g"]  = round(std, 6)
            if not checks["8_non_constant"]:
                errors.append(
                    f"Signal is constant (std={std:.2e} g). "
                    "Sensor may be stuck or not properly initialised."
                )
        else:
            checks["8_non_constant"] = False

        # ── Check 9: Within physical range ────────────────────────────────────
        if x_samples:
            arr      = np.array(x_samples)
            headroom = range_g * 1.20   # 20% above nominal range
            in_range = bool(np.all(np.abs(arr) <= headroom))
            checks["9_physical_range"] = in_range
            details["x_max_g"] = round(float(np.max(np.abs(arr))), 4)
            details["range_limit_g"] = headroom
            if not in_range:
                warnings.append(
                    f"Samples exceed ±{headroom:.1f} g "
                    f"(configured range ±{range_g}g × 1.2 headroom). "
                    "Check sensor orientation and mounting. "
                    "Gravity (~1g) should appear on the vertical axis."
                )
        else:
            checks["9_physical_range"] = False

    finally:
        # ── Check 10: Return to standby ────────────────────────────────────────
        try:
            bus.write_byte_data(address, REG_POWER_CTL, 0x00)
            checks["10_standby_ok"] = True
        except Exception as exc:
            checks["10_standby_ok"] = False
            warnings.append(f"Could not set standby mode: {exc}")

        if locals().get("own_bus"):
            try:
                bus.close()
            except Exception:
                pass

    passed = all(checks.values())
    return VerificationResult(
        passed   = passed,
        checks   = checks,
        errors   = errors,
        warnings = warnings,
        details  = details,
    )


# ─── CLI ENTRY POINT ──────────────────────────────────────────────────────────

def _main() -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Verify ADXL345 sensor connectivity and operation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m edge.sensors.adxl345_verify
  python -m edge.sensors.adxl345_verify --bus 1 --address 0x1D
  python -m edge.sensors.adxl345_verify --json > result.json
        """,
    )
    parser.add_argument("--bus",     type=int, default=DEFAULT_BUS,
                        help=f"I2C bus number (default: {DEFAULT_BUS})")
    parser.add_argument("--address", type=lambda x: int(x, 16), default=DEFAULT_ADDRESS,
                        help=f"I2C address in hex (default: 0x{DEFAULT_ADDRESS:02X})")
    parser.add_argument("--range",   type=int, default=2, choices=[2, 4, 8, 16],
                        help="Expected accelerometer range in g (default: 2)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of test samples (default: 10)")
    parser.add_argument("--quiet",   action="store_true",
                        help="Suppress output (only exit code)")
    parser.add_argument("--json",    action="store_true",
                        help="Output result as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    result = verify_adxl345(
        bus_number     = args.bus,
        address        = args.address,
        n_test_samples = args.samples,
        range_g        = args.range,
    )

    if args.json:
        print(json.dumps({
            "passed":   result.passed,
            "checks":   result.checks,
            "errors":   result.errors,
            "warnings": result.warnings,
            "details":  result.details,
        }, indent=2))
    elif not args.quiet:
        print(f"\nAuraPredict — ADXL345 Hardware Verification")
        print(f"Bus: {args.bus} | Address: 0x{args.address:02X} | Range: ±{args.range}g\n")
        for name, ok in result.checks.items():
            icon = "✅" if ok else "❌"
            label = name.split("_", 1)[1].replace("_", " ").title()
            print(f"  {icon} {label}")
        if result.warnings:
            print("\nWarnings:")
            for w in result.warnings:
                print(f"  ⚠️  {w}")
        if result.errors:
            print("\nErrors:")
            for e in result.errors:
                print(f"  ❌ {e}")
        print(f"\n{result.summary()}\n")

    return 0 if result.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(_main())
