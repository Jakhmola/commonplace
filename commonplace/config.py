from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://commonplace:commonplace@localhost:5433/commonplace"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    inngest_dev: bool = True
    inngest_base_url: str = "http://localhost:8288"
    app_port: int = 8000
    user_agent: str = "commonplace/0.1"


settings = Settings()
