"""
Authentication Pydantic Schemas
"""
from datetime import datetime
import re
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class UserSignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="Unique email address")
    password: str = Field(..., min_length=6, max_length=128, description="User password")
    confirm_password: str = Field(..., min_length=6, max_length=128, description="Password confirmation")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", cleaned):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens.")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.strip().lower()

    @model_validator(mode="after")
    def check_passwords_match(self) -> "UserSignupRequest":
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match.")
        return self


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
