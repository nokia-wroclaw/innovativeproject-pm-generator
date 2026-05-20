"""Integration layer with Apache Airflow REST API v2.

Modules:
    - config:  loads Airflow connection settings from env
    - auth:    mints and caches the service-account JWT used to call Airflow
    - client:  async httpx client wrapper (raw HTTP, no domain mapping)
    - mapper:  translates Airflow API v2 payloads into our Pydantic DTOs
    - errors:  exceptions raised by the integration layer
"""
