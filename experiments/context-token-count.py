"""
Token 计数练习 — 理解上下文窗口

运行：python experiments/context-token-count.py

它会演示：
  1. 不同模型编码器的 token 计数差异
  2. 中英文混合文本的 token 占用
  3. 一个简单的上下文窗口填充模拟
"""

try:
    import tiktoken
except ImportError:
    print("❌ 需要安装 tiktoken，运行: pip install tiktoken")
    exit(1)


def demo_encodings():
    """不同编码器对同一段文本的 token 数"""
    text = "Hello, AI × Web3 School! 你好，AI × Web3 学校！"

    encodings = [
        ("cl100k_base", "GPT-3.5/4 (8K/32K)"),
        ("o200k_base", "GPT-4o / o1 / o3"),
    ]

    print("=" * 55)
    print("🌐 不同模型编码器的 token 数对比")
    print("=" * 55)
    print(f"文本: {text}\n")

    for enc_name, model_desc in encodings:
        enc = tiktoken.get_encoding(enc_name)
        tokens = enc.encode(text)
        print(f"  {model_desc} ({enc_name})")
        print(f"    → {len(tokens)} tokens")
        print(f"    前10个 token IDs: {tokens[:10]}")
    print()


def demo_chinese_vs_english():
    """中英文 token 计数对比"""
    english = "Hello, this is a test sentence in English to compare token counts between languages."
    chinese = "你好，这是一句中文测试句子，用来对比中英文之间的 token 数量差异。"

    enc = tiktoken.get_encoding("cl100k_base")

    print("=" * 55)
    print("🇨🇳 中英文 token 计数对比 (cl100k_base)")
    print("=" * 55)

    for label, text in [("English", english), ("中文", chinese)]:
        tokens = enc.encode(text)
        chars = len(text)
        print(f"\n  [{label}]")
        print(f"    字符数: {chars}")
        print(f"    Token 数: {len(tokens)}")
        print(f"    平均 chars/token: {chars / len(tokens):.1f}")


def simulate_context_fill():
    """模拟上下文窗口填充"""
    enc = tiktoken.get_encoding("cl100k_base")
    window_size = 4096  # 模拟一个小窗口的模型

    print("\n" + "=" * 55)
    print("📊 模拟上下文窗口填充 (GPT-3.5, 4K 窗口)")
    print("=" * 55)
    print(f"窗口上限: {window_size} tokens\n")

    items = [
        ("系统指令", "你是一个 Web3 交易助手续……", 180),
        ("工具定义", "可用工具：get_balance, send_transaction, sign_message……", 320),
        ("用户输入", "帮我把 0.5 ETH 转到 0x1234……", 50),
        ("链上数据", "地址交易历史、余额、模拟结果……", 1200),
        ("对话历史", "前几轮的问答记录……", 800),
    ]

    used = 0
    for name, desc, tokens in items:
        used += tokens
        remain = window_size - used
        bar_len = 30
        fill = max(0, int(bar_len * used / window_size))
        bar = "█" * fill + "░" * (bar_len - fill)

        if used <= window_size:
            print(f"  ✅ {bar} {name:10s} +{tokens:>4}t | 已用 {used:>4}t 剩余 {remain:>4}t")
        else:
            print(f"  ❌ {bar} {name:10s} +{tokens:>4}t | 💥 超出窗口 {abs(remain):>4}t")

    print()
    print(f"  重要观察：链上数据 + 对话历史 ≈ 2000 tokens")
    print(f"  → 在一个 4K 窗口里，留给实际计算的空间不多了")
    print(f"  → 这就是为什么 Agent 需要 上下文压缩 和 RAG")


if __name__ == "__main__":
    demo_encodings()
    demo_chinese_vs_english()
    simulate_context_fill()
