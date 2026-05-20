from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import HttpUrl, SecretStr
from typing import Optional


class Settings(BaseSettings):
    s3_url: HttpUrl
    s3_external_url: HttpUrl
    s3_bucket: str

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


settings = Settings()
