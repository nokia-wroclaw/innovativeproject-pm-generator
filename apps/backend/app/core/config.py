from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    s3_url: HttpUrl = Field(validation_alias="S3_URL")
    s3_external_url: HttpUrl = Field(validation_alias="S3_EXTERNAL_URL")
    s3_bucket: str = Field(validation_alias="S3_BUCKET")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


settings = Settings()  # type: ignore[call-arg]
