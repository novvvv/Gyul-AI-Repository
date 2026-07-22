import pytest

from app.core.security import TokenError, verify_access_token
from tests.conftest import make_access_token


def test_valid_token_returns_claims(jwt_secret):
    token = make_access_token()

    payload = verify_access_token(token)

    assert payload["sub"] == "ai-test@example.com"
    assert payload["memberId"] == 42
    assert payload["role"] == "MEMBER"


def test_forged_token_rejected(jwt_secret):
    token = make_access_token()
    forged = token[:-2] + ("aa" if not token.endswith("aa") else "bb")

    with pytest.raises(TokenError) as exc:
        verify_access_token(forged)
    assert exc.value.reason == "invalid"


def test_wrong_secret_rejected(jwt_secret):
    token = make_access_token(secret="another-secret-value-32bytes-padding!!")

    with pytest.raises(TokenError) as exc:
        verify_access_token(token)
    assert exc.value.reason == "invalid"


def test_expired_token_rejected(jwt_secret):
    # leeway(10초)보다 확실히 과거로 만료
    token = make_access_token(expires_in=-60)

    with pytest.raises(TokenError) as exc:
        verify_access_token(token)
    assert exc.value.reason == "expired"


def test_alg_none_rejected(jwt_secret):
    token = make_access_token(secret="", algorithm="none")

    with pytest.raises(TokenError) as exc:
        verify_access_token(token)
    assert exc.value.reason == "invalid"


def test_refresh_style_token_without_member_id_rejected(jwt_secret):
    # Refresh Token에는 memberId/role claim이 없다 (가이드 §1.2)
    token = make_access_token(memberId=None, role=None)

    with pytest.raises(TokenError) as exc:
        verify_access_token(token)
    assert exc.value.reason == "invalid"


def test_missing_sub_rejected(jwt_secret):
    token = make_access_token(sub=None)

    with pytest.raises(TokenError):
        verify_access_token(token)


def test_missing_secret_rejects_all(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config, "JWT_SECRET", "")

    with pytest.raises(TokenError):
        verify_access_token(make_access_token())
