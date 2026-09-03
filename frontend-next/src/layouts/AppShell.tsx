import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { Character } from "../components/Character";

const NAV = [
  { to: "/", label: "소개", end: true },
  { to: "/demo", label: "대화", end: false },
  { to: "/demo/report", label: "리포트", end: false },
  { to: "/my", label: "내 기록", end: false },
];

export function AppShell() {
  const { pathname } = useLocation();
  const onLogin = pathname === "/login";

  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">
          결
        </Link>

        {!onLogin && (
          <nav className="topnav">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) => (isActive ? "on" : "")}
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        )}

        {onLogin ? (
          <div className="acct-chip" style={{ marginLeft: "auto" }}>
            <Link className="btn sm" to="/login">
              로그인
            </Link>
          </div>
        ) : (
          <div className="acct-chip">
            <span className="av2">
              <Character kind="gyul" mood="idle" />
            </span>
            <span className="em">nov@gyul.kr</span>
          </div>
        )}
      </header>

      <Outlet />
    </div>
  );
}
