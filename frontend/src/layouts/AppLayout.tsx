import { Link, NavLink, Outlet } from "react-router-dom";
import "../styles/layout.css";

export function AppLayout() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <Link to="/" className="brand">
          <span className="logo">결</span>
          <span className="brand-text">
            <strong>결</strong>
            <small>AI 음성 자기 탐색</small>
          </span>
        </Link>
        <nav>
          <NavLink to="/" end>
            소개
          </NavLink>
          <NavLink to="/demo">체험하기</NavLink>
        </nav>
      </header>
      <Outlet />
      <footer className="site-footer">
        <span>© 결 — 목소리로 읽는 자기 탐색</span>
      </footer>
    </div>
  );
}
