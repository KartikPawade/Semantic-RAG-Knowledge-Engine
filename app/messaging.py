"""
Background workers: RabbitMQ messaging for ingestion tasks.

A file lands in storage → a message is published → a worker picks it up →
parsed, chunked, tagged, vectors pushed to ChromaDB. Keeps the API responsive
and isolates heavy/crashing jobs from the rest of the system.
"""
import json
import uuid

import pika
from pydantic import BaseModel


class IngestTaskPayload(BaseModel):
    task_id: str
    file_path: str
    filename: str


def publish_ingest_task(
    file_path: str,
    filename: str,
    task_id: str | None = None,
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
    queue_name: str = "ingestion_tasks",
) -> str:
    """
    Publish one ingestion task to RabbitMQ. Returns task_id.
    Worker will: read file → hash → idempotency check → parse/chunk/embed → record hash.
    """
    task_id = task_id or uuid.uuid4().hex
    payload = IngestTaskPayload(task_id=task_id, file_path=file_path, filename=filename)
    body = payload.model_dump_json()
    params = pika.URLParameters(rabbitmq_url)
    conn = pika.BlockingConnection(params)
    try:
        ch = conn.channel()
        ch.queue_declare(queue=queue_name, durable=True)
        ch.basic_publish(
            exchange="",
            routing_key=queue_name,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2),
        )
        return task_id
    finally:
        conn.close()


def consume_ingest_tasks(
    callback,
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/",
    queue_name: str = "ingestion_tasks",
) -> None:
    """
    Consume messages from the ingestion queue and call callback(body_dict) for each.
    Blocks until connection is closed. callback should ack/nack the message.
    """
    params = pika.URLParameters(rabbitmq_url)
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.queue_declare(queue=queue_name, durable=True)
    ch.basic_qos(prefetch_count=1)

    def _on_message(channel, method, properties, body):
        try:
            data = json.loads(body)
            callback(data, channel, method)
        except Exception:
            channel.basic_nack(method.delivery_tag, requeue=False)

    ch.basic_consume(queue=queue_name, on_message_callback=_on_message)
    ch.start_consuming()
