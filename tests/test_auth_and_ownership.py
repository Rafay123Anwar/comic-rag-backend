"""
Comprehensive Automated Test Suite for Auth, Comic Ownership, and Protected Endpoints
"""
import uuid
import pytest
import httpx

from app.core.database import init_db
from app.main import app
from app.services.storage import delete_comic_storage, save_comic_json


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    init_db()
    yield


@pytest.fixture
async def async_client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_auth_full_flow(async_client):
    unique_suffix = uuid.uuid4().hex[:8]
    username = f"user_{unique_suffix}"
    email = f"{username}@example.com"
    password = "SuperSecretPassword123!"

    # 1. Test Password Mismatch
    mismatch_resp = await async_client.post("/auth/signup", json={
        "username": username,
        "email": email,
        "password": password,
        "confirm_password": "WrongPassword123!"
    })
    assert mismatch_resp.status_code == 422, mismatch_resp.text

    # 2. Successful Signup
    signup_resp = await async_client.post("/auth/signup", json={
        "username": username,
        "email": email,
        "password": password,
        "confirm_password": password
    })
    assert signup_resp.status_code == 201, signup_resp.text
    signup_data = signup_resp.json()
    assert signup_data["username"] == username
    assert signup_data["email"] == email
    assert "hashed_password" not in signup_data
    assert "id" in signup_data

    # 3. Duplicate Username
    dup_user_resp = await async_client.post("/auth/signup", json={
        "username": username,
        "email": f"diff_{email}",
        "password": password,
        "confirm_password": password
    })
    assert dup_user_resp.status_code == 400, dup_user_resp.text
    assert "username already exists" in dup_user_resp.text.lower()

    # 4. Duplicate Email
    dup_email_resp = await async_client.post("/auth/signup", json={
        "username": f"diff_{username}",
        "email": email,
        "password": password,
        "confirm_password": password
    })
    assert dup_email_resp.status_code == 400, dup_email_resp.text
    assert "email address already exists" in dup_email_resp.text.lower()

    # 5. Invalid Login Credentials
    bad_login_resp = await async_client.post("/auth/login", json={
        "email": email,
        "password": "WrongPassword"
    })
    assert bad_login_resp.status_code == 401, bad_login_resp.text

    # 6. Successful Login
    login_resp = await async_client.post("/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_resp.status_code == 200, login_resp.text
    login_data = login_resp.json()
    token = login_data["access_token"]
    assert token
    assert login_data["token_type"] == "bearer"
    assert login_data["user"]["email"] == email

    # 7. Access /auth/me Unauthenticated
    unauth_me = await async_client.get("/auth/me")
    assert unauth_me.status_code == 401

    # 8. Access /auth/me Authenticated
    auth_me = await async_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert auth_me.status_code == 200, auth_me.text
    me_data = auth_me.json()
    assert me_data["id"] == signup_data["id"]
    assert me_data["username"] == username
    assert me_data["email"] == email
    assert "hashed_password" not in me_data


@pytest.mark.anyio
async def test_comic_ownership_isolation(async_client):
    # Create User A
    suffix_a = uuid.uuid4().hex[:6]
    email_a = f"usera_{suffix_a}@example.com"
    await async_client.post("/auth/signup", json={
        "username": f"usera_{suffix_a}",
        "email": email_a,
        "password": "Password123!",
        "confirm_password": "Password123!"
    })
    res_a = await async_client.post("/auth/login", json={"email": email_a, "password": "Password123!"})
    token_a = res_a.json()["access_token"]

    # Create User B
    suffix_b = uuid.uuid4().hex[:6]
    email_b = f"userb_{suffix_b}@example.com"
    await async_client.post("/auth/signup", json={
        "username": f"userb_{suffix_b}",
        "email": email_b,
        "password": "Password123!",
        "confirm_password": "Password123!"
    })
    res_b = await async_client.post("/auth/login", json={"email": email_b, "password": "Password123!"})
    token_b = res_b.json()["access_token"]

    # Get User IDs
    me_a = await async_client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    user_a_id = me_a.json()["id"]

    # Create a dummy comic owned by User A
    comic_id = str(uuid.uuid4())
    save_comic_json(
        comic_id=comic_id,
        comic_name="User A Secret Comic",
        source_format="cbz",
        pages=[{
            "page_number": 1,
            "filename": "page_001.jpg",
            "image_path": "page_001.jpg",
            "analysis": {"text": {"full_text": "Secret User A Text"}},
            "status": "success"
        }],
        status="completed",
        total_pages=1,
        user_id=user_a_id
    )

    try:
        # 1. User A should see the comic in library
        list_a_resp = await async_client.get("/comics", headers={"Authorization": f"Bearer {token_a}"})
        list_a = list_a_resp.json()
        assert any(c["comic_id"] == comic_id for c in list_a)

        # 2. User B should NOT see User A's comic in library
        list_b_resp = await async_client.get("/comics", headers={"Authorization": f"Bearer {token_b}"})
        list_b = list_b_resp.json()
        assert not any(c["comic_id"] == comic_id for c in list_b)

        # 3. User A can get comic details & status
        detail_a = await async_client.get(f"/comics/{comic_id}", headers={"Authorization": f"Bearer {token_a}"})
        assert detail_a.status_code == 200
        status_a = await async_client.get(f"/comics/{comic_id}/status", headers={"Authorization": f"Bearer {token_a}"})
        assert status_a.status_code == 200

        # 4. User B CANNOT get User A's comic details or status (404)
        detail_b = await async_client.get(f"/comics/{comic_id}", headers={"Authorization": f"Bearer {token_b}"})
        assert detail_b.status_code == 404
        status_b = await async_client.get(f"/comics/{comic_id}/status", headers={"Authorization": f"Bearer {token_b}"})
        assert status_b.status_code == 404

        # 5. User B CANNOT ask questions about User A's comic
        ask_b = await async_client.post(
            f"/comics/{comic_id}/ask",
            json={"question": "What is the secret?"},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert ask_b.status_code == 404

        # 6. User A can create conversation on their comic
        conv_create = await async_client.post(
            "/conversations",
            json={"comic_id": comic_id},
            headers={"Authorization": f"Bearer {token_a}"}
        )
        assert conv_create.status_code == 200
        conv_id = conv_create.json()["conversation_id"]

        # 7. User B CANNOT access User A's conversation
        conv_b_get = await async_client.get(
            f"/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert conv_b_get.status_code == 404

        # 8. User B CANNOT create conversation for User A's comic
        conv_b_create = await async_client.post(
            "/conversations",
            json={"comic_id": comic_id},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert conv_b_create.status_code == 404

        # 9. Unauthenticated image access returns 401
        img_unauth = await async_client.get(f"/comics/{comic_id}/pages/1/image")
        assert img_unauth.status_code == 401

        # 10. User B cannot access User A's comic image (404)
        img_user_b = await async_client.get(
            f"/comics/{comic_id}/pages/1/image",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert img_user_b.status_code == 404

        # 11. Unauthenticated streaming chat returns 401
        stream_unauth = await async_client.post(
            f"/conversations/{conv_id}/stream",
            json={"question": "Tell me about this comic"}
        )
        assert stream_unauth.status_code == 401

        # 12. User B cannot stream on User A's conversation (404)
        stream_user_b = await async_client.post(
            f"/conversations/{conv_id}/stream",
            json={"question": "Tell me about this comic"},
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert stream_user_b.status_code == 404

        # 13. Unauthenticated thumbnail request returns 401
        thumb_unauth = await async_client.get(f"/comics/{comic_id}/pages/1/thumbnail")
        assert thumb_unauth.status_code == 401

        # 14. User B cannot access User A's thumbnail (404)
        thumb_user_b = await async_client.get(
            f"/comics/{comic_id}/pages/1/thumbnail",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        assert thumb_user_b.status_code == 404

    finally:
        delete_comic_storage(comic_id)
