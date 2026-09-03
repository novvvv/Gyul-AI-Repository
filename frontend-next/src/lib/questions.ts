/**
 * 고정 문항 — 자가분석 4문항 · 면접 4문항.
 *
 * 두 단계는 **같은 네 가지 주제**를 다루되 프레임만 바꾼다.
 * 그래야 "편할 때의 나 / 면접에서의 나"를 같은 축에서 비교할 수 있다.
 *
 *   주제        자가분석                  면접
 *   ──────────────────────────────────────────────────
 *   현재        요즘 어떻게 지내는지       자기소개
 *   성취        잘했다고 느낀 일          성과가 뚜렷했던 일
 *   관계        마음에 걸리는 일          의견이 갈렸던 경험
 *   방향        1년 뒤 바라는 모습        실패와 그 이후
 *
 * 말투는 두 상대가 확연히 달라야 한다.
 * 결은 친구에게 묻듯 짧고 편하게, 면접관은 격식을 지킨다.
 * 양쪽 모두 1~2문장을 넘기지 않는다. 길어지면 대본처럼 들린다.
 */

export type Stage = "self" | "interview";

export type Question = {
  id: string;
  /** 두 단계에서 짝이 되는 주제 키 */
  topic: "present" | "achievement" | "relation" | "direction";
  topicName: string;
  /** 화면에 뜨는 질문 — 1~2문장 */
  text: string;
};

export const SELF_QUESTIONS: Question[] = [
  {
    id: "s1",
    topic: "present",
    topicName: "지금의 나",
    text: "요즘 어떻게 지내세요?",
  },
  {
    id: "s2",
    topic: "achievement",
    topicName: "잘한 일",
    text: "최근에 뿌듯했던 일 있었어요?",
  },
  {
    id: "s3",
    topic: "relation",
    topicName: "마음에 걸리는 일",
    text: "반대로 마음에 걸리는 일은요?",
  },
  {
    id: "s4",
    topic: "direction",
    topicName: "바라는 방향",
    text: "1년 뒤엔 어떻게 지내고 싶어요?",
  },
];

export const INTERVIEW_QUESTIONS: Question[] = [
  {
    id: "i1",
    topic: "present",
    topicName: "자기소개",
    text: "그럼 시작하겠습니다. 간단히 자기소개 부탁드립니다.",
  },
  {
    id: "i2",
    topic: "achievement",
    topicName: "성과 경험",
    text: "성과가 가장 뚜렷했던 일 하나 말씀해 주시겠어요?",
  },
  {
    id: "i3",
    topic: "relation",
    topicName: "협업·갈등",
    text: "일하다 의견이 갈렸던 경험이 있으신가요?",
  },
  {
    id: "i4",
    topic: "direction",
    topicName: "실패와 개선",
    text: "마지막으로, 뜻대로 안 됐던 경험을 말씀해 주세요.",
  },
];

export function questionsFor(stage: Stage): Question[] {
  return stage === "self" ? SELF_QUESTIONS : INTERVIEW_QUESTIONS;
}

/** 같은 주제의 반대편 문항 — 리포트에서 짝지어 보여줄 때 쓴다 */
export function counterpart(q: Question, stage: Stage): Question | undefined {
  const other = stage === "self" ? INTERVIEW_QUESTIONS : SELF_QUESTIONS;
  return other.find((x) => x.topic === q.topic);
}
