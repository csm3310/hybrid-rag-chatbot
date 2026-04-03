document.getElementById("sendBtn").addEventListener("click", ask);

function appendMessage(text, sender) {
    const chatWindow = document.getElementById("chat-window");

    const msg = document.createElement("div");
    msg.classList.add("message");
    msg.classList.add(sender === "user" ? "user-message" : "bot-message");
    msg.innerHTML = text;

    chatWindow.appendChild(msg);
    chatWindow.scrollTop = chatWindow.scrollHeight;  // 자동 스크롤
}

// 봇 타이핑 애니메이션 전용 함수
function appendBotTyping(text) {
    const chatWindow = document.getElementById("chat-window");

    // 빈 말풍선 먼저 생성
    const msg = document.createElement("div");
    msg.classList.add("message");
    msg.classList.add("bot-message");
    chatWindow.appendChild(msg);

    let i = 0;

    function type() {
        if (i < text.length) {
            msg.innerText += text.charAt(i);
            i++;
            chatWindow.scrollTop = chatWindow.scrollHeight;  // 계속 맨 아래로
            setTimeout(type, 15);  // 숫자 줄이면 더 빨라짐 (타이핑 속도)
        }
    }

    type();
}

async function ask() {
    const q = document.getElementById("question").value.trim();
    if (!q) return;

    // 사용자 메시지를 채팅창에 표시
    appendMessage(q, "user");

    // 입력창 초기화
    document.getElementById("question").value = "";

    // API 호출
    const res = await fetch("http://127.0.0.1:5001/ask", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ query: q })
    }).then(r => r.json());

    // 봇 답변 출력
    // 답변만 타이핑 애니메이션으로 출력
    appendBotTyping(res.answer);

    // 버튼은 타이핑 완료 이후 약간의 딜레이 후 렌더링
    if (res.url) {
        setTimeout(() => {
            appendMessage(`
                <a href="${res.url}" target="_blank"
                style="display:inline-block; margin-top:6px; padding:8px 14px;
                    background:#007bff; color:#fff; border-radius:8px;
                    text-decoration:none; font-weight:600;">
                🔗 자세히 보기
                </a>
            `, "bot");
        }, res.answer.length * 15 + 300);   // 타이핑 길이 따라 timing 조정
    }

    }


