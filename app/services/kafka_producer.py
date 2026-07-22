"""Kafka 분석 리포트 발행 (연동 가이드 §3).

topic interview.analysis-report, key=email(UTF-8), value=스키마 v1 JSON.
Spring이 sessionId로 멱등 처리하므로 at-least-once로 편하게 재시도한다.
브로커 다운이 모델 서빙을 막지 않도록 실패는 로그로만 남기고 서버는 계속 뜬다.
"""

import asyncio
import json
import logging

from app.core.config import (
    ENABLE_KAFKA,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_REPORT_TOPIC,
)

logger = logging.getLogger(__name__)

RETRY_BACKOFF_SECONDS = 2.0


class ReportKafkaProducer:
    def __init__(self) -> None:
        self._producer = None

    @property
    def is_enabled(self) -> bool:
        return ENABLE_KAFKA

    async def start(self) -> None:
        if not ENABLE_KAFKA or self._producer is not None:
            return
        try:
            from aiokafka import AIOKafkaProducer

            producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                acks="all",
                enable_idempotence=True,
            )
            await producer.start()
            self._producer = producer
            logger.info("Kafka producer 연결: %s", KAFKA_BOOTSTRAP_SERVERS)
        except Exception as e:
            # startup 실패로 서버가 죽으면 안 됨 — 발행 시점에 lazy 재시도
            self._producer = None
            logger.warning("Kafka producer 시작 실패(발행 시 재시도): %s", e)

    async def stop(self) -> None:
        if self._producer is not None:
            try:
                await self._producer.stop()
            finally:
                self._producer = None

    async def publish_report(self, report: dict) -> bool:
        """스키마 v1 리포트 발행. 성공 True / 최종 실패 False.

        실패 시 백오프 후 1회 재시도하고, 그래도 실패하면 수동 재발행이
        가능하도록 리포트 전문을 ERROR 로그로 남긴다 (sessionId 멱등이라 안전).
        """
        if not ENABLE_KAFKA:
            logger.info(
                "ENABLE_KAFKA=0 — 리포트 발행 생략 (sessionId=%s)",
                report.get("sessionId"),
            )
            return False

        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                if self._producer is None:
                    await self.start()
                if self._producer is None:
                    raise RuntimeError("Kafka producer unavailable")

                await self._producer.send_and_wait(
                    KAFKA_REPORT_TOPIC,
                    key=report["email"].encode("utf-8"),
                    value=json.dumps(report, ensure_ascii=False).encode("utf-8"),
                )
                logger.info(
                    "분석 리포트 발행 완료 (sessionId=%s)", report.get("sessionId")
                )
                return True
            except Exception as e:
                last_error = e
                if attempt == 1:
                    await asyncio.sleep(RETRY_BACKOFF_SECONDS)

        logger.error(
            "분석 리포트 발행 최종 실패 (sessionId=%s): %s — 수동 재발행용 payload: %s",
            report.get("sessionId"),
            last_error,
            json.dumps(report, ensure_ascii=False),
        )
        return False


report_kafka_producer = ReportKafkaProducer()
