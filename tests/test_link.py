import random

import pytest

from turret.link import cobs
from turret.link import messages as m
from turret.link.crc import crc16
from turret.link.framing import FrameDecoder, encode_frame
from turret.link.link import Link
from turret.link.transport import sim_wire


def test_crc_check_value():
    assert crc16(b"123456789") == 0x29B1


@pytest.mark.parametrize("n", [0, 1, 2, 253, 254, 255, 256, 600])
def test_cobs_roundtrip_lengths(n):
    rng = random.Random(n)
    data = bytes(rng.choice([0, 0, 1, 7, 255, rng.randrange(256)]) for _ in range(n))
    enc = cobs.encode(data)
    assert 0 not in enc
    assert cobs.decode(enc) == data


def test_cobs_fuzz():
    rng = random.Random(1)
    for _ in range(2000):
        data = bytes(rng.randrange(256) for _ in range(rng.randrange(0, 300)))
        assert cobs.decode(cobs.encode(data)) == data


def test_every_message_roundtrips():
    for cls in m.BY_ID.values():
        msg = cls()
        assert len(msg.pack()) == cls.SIZE
        frame = encode_frame(cls.ID, msg.pack())
        out = list(FrameDecoder().feed(frame))
        assert out == [(cls.ID, msg.pack())]
        assert cls.unpack(out[0][1]) == msg


def test_decoder_survives_split_reads_garbage_and_corruption():
    t = m.Traj(valid_at_us=123, yaw_deg=1.5, mode=2)
    good = encode_frame(t.ID, t.pack())
    bad = bytearray(good)
    bad[2] ^= 0x10
    stream = b"\x13\x37\x00" + bytes(bad) + good + good
    dec = FrameDecoder()
    got = []
    for i in range(0, len(stream), 3):  # serial reads never align with packets
        got += list(dec.feed(stream[i : i + 3]))
    assert len(got) == 2
    assert m.Traj.unpack(got[0][1]) == t
    assert dec.stats.crc_errors + dec.stats.cobs_errors >= 2


def test_string_field_truncates_and_strips():
    msg = m.Log(level=2, text="x" * 100)
    back = m.Log.unpack(msg.pack())
    assert back.text == "x" * 48
    assert m.Log.unpack(m.Log(text="hi").pack()).text == "hi"


class FakeEsp:
    """Minimal responder for handshake tests."""

    def __init__(self, end, now, proto_hash):
        self.end, self.now, self.hash = end, now, proto_hash
        self.dec = FrameDecoder()
        self.got_config = False

    def poll(self):
        for mid, payload in self.dec.feed(self.end.read()):
            if mid == m.Hello.ID:
                ack = m.HelloAck(proto_hash=self.hash, fw_version=3)
                self.end.write(encode_frame(ack.ID, ack.pack()))
            elif mid == m.TimePing.ID:
                seq = m.TimePing.unpack(payload).seq
                t = int((self.now() + 50.0) * 1e6)
                pong = m.TimePong(seq=seq, t_rx_us=t, t_tx_us=t + 20)
                self.end.write(encode_frame(pong.ID, pong.pack()))
            elif mid == m.Config.ID:
                self.got_config = True


def _run(proto_hash, seconds=2.0):
    clock = {"t": 10.0}

    def now():
        return clock["t"]

    j, e = sim_wire(now, 0.001, 0.0005)
    link = Link(j, now, config_msg=m.Config(trigger_hz=90.0))
    esp = FakeEsp(e, now, proto_hash)
    while clock["t"] < 10.0 + seconds:
        clock["t"] += 0.001
        link.poll()
        esp.poll()
    return link, esp


def test_handshake_config_and_clock_sync():
    link, esp = _run(m.PROTO_HASH)
    assert link.connected and link.synced
    assert esp.got_config
    assert link.stats.fw_version == 3
    # Offset is exactly +50 s; symmetric-ish jitter keeps the estimate well under 1 ms.
    assert abs(link.clock.offset_at(11.0) - 50.0) < 0.0005


def test_firmware_with_wrong_protocol_hash_is_refused():
    link, esp = _run(m.PROTO_HASH ^ 1)
    assert not link.connected
    assert link.stats.hash_mismatch
    assert not esp.got_config
