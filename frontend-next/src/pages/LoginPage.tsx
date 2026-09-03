import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Character } from "../components/Character";

/**
 * 로그인 — 인증은 Spring 서버 담당(JWT HS256).
 * 아직 연동 전이라 제출하면 그대로 통과시킨다.
 * 실제 연동 시 여기서 토큰을 받아 저장하고, WS 접속에 `?token=` 으로 실어 보낸다.
 */
export function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("nov@gyul.kr");
  const [pw, setPw] = useState("");

  const submit = (e: FormEvent) => {
    e.preventDefault();
    navigate("/my");
  };

  return (
    <main className="auth">
      <div className="auth-side brand-side">
        <span className="auth-fig">
          <Character kind="gyul" mood="listen" />
        </span>
        <h2>
          편할 때의 나와
          <br />
          면접에서의 나
        </h2>
        <p className="lede">
          두 번의 대화로 그 사이의 간격을 봅니다. 기록은 계정에 남고, 다음 면접 전에 다시
          열어볼 수 있습니다.
        </p>
        <div className="marks">
          <div>
            <span className="n">01</span>
            <span>자가분석 대화로 기준선을 만듭니다</span>
          </div>
          <div>
            <span className="n">02</span>
            <span>같은 주제를 면접에서 다시 묻습니다</span>
          </div>
          <div>
            <span className="n">03</span>
            <span>여섯 축의 차이를 한 장으로 받습니다</span>
          </div>
        </div>
      </div>

      <div className="seam" />

      <div className="auth-side">
        <form className="form" onSubmit={submit}>
          <h1>로그인</h1>
          <p className="sub">이어서 대화하거나 지난 기록을 확인합니다.</p>

          <div className="field">
            <label htmlFor="lg-em">이메일</label>
            <input
              id="lg-em"
              type="email"
              autoComplete="username"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="lg-pw">비밀번호</label>
            <input
              id="lg-pw"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••"
              value={pw}
              onChange={(e) => setPw(e.target.value)}
            />
          </div>

          <div className="row2">
            <label>
              <input type="checkbox" defaultChecked /> 로그인 유지
            </label>
            <span style={{ color: "var(--ink-3)" }}>비밀번호 찾기</span>
          </div>

          <button className="btn solid block" type="submit">
            로그인
          </button>

          <p className="alt">계정이 없으신가요? 회원가입</p>
          <p className="note">인증은 Spring 서버가 처리합니다 · JWT HS256</p>
        </form>
      </div>
    </main>
  );
}
