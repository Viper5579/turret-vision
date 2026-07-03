# ESP32 Wire Protocol (reference for future firmware)

Authoritative definition lives in SPEC.md section 5. Summary for the firmware author (you, later):

- Binary, little-endian. Frame: `AA 55 | LEN u8 | TYPE u8 | PAYLOAD | CRC16 u16`
- CRC-16/CCITT-FALSE over LEN..PAYLOAD. Reject frame on mismatch, resync on next AA 55.
- Parse byte-at-a-time with a state machine. Never assume a read() aligns to a packet.
- TYPE 0x01 AIM (Jetson->ESP32, ~50 Hz): t_ms u32, flags u8 (bit0 target_valid,
  bit1 e-stop, bit2-3 mode), yaw_set i16 centideg ABSOLUTE, pitch_set i16,
  yaw_rate i16 cdps, pitch_rate i16, confidence u8, range_mm u16.
- TYPE 0x02 TELEM (ESP32->Jetson, ~100 Hz): t_ms u32 (millis), yaw_pos i16 centideg,
  pitch_pos i16, yaw_rate i16, status u8 (bit0 homed, bit1 yaw@limit, bit2 pitch@limit,
  bit3 fault, bit4 e-stopped).
- Firmware MUST: clamp setpoints to travel limits (never trust the wire), stop motion
  while e-stop bit set, enter safe-hold if no valid AIM frame for a timeout (link-dead
  is distinguishable from no-target BECAUSE the Jetson heartbeats target_valid=0 packets).
- Jetson-side pack/unpack reference implementation arrives in Phase 3 as
  turretvision/link/protocol.py -- port the constants from there, do not retype them.
