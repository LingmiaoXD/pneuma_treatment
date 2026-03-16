# 简单的F1-score计算器
# 手动输入精准率(P)和召回率(R)，计算并打印F1-score

def calculate_f1_score(precision, recall):
    """
    计算F1-score
    F1 = 2 * (P * R) / (P + R)
    """
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

if __name__ == "__main__":
    print("=" * 50)
    print("F1-score 计算器")
    print("=" * 50)
    
    # 获取用户输入
    precision = float(input("请输入精准率 (P): "))
    recall = float(input("请输入召回率 (R): "))
    
    # 计算F1-score
    f1_score = calculate_f1_score(precision, recall)
    
    # 打印结果
    print("\n" + "=" * 50)
    print(f"精准率 (P): {precision:.4f}")
    print(f"召回率 (R): {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    print("=" * 50)
