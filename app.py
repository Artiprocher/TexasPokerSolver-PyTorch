import torch
from tqdm import tqdm
import streamlit as st
import pandas as pd


class 德州扑克求解器(torch.nn.Module):
    def __init__(self, 人数, 扑克牌数, dtype=torch.int32, device="cuda"):
        super().__init__()
        self.人数 = 人数
        self.扑克牌数 = 扑克牌数
        self.dtype = dtype
        self.device = device
        self.eps = 1e-1
        self.花色字典 = {
            "黑桃": 0,
            "红桃": 1,
            "梅花": 2,
            "方片": 3
        }
        self.数字字典 = {
            "2": 0,
            "3": 1,
            "4": 2,
            "5": 3,
            "6": 4,
            "7": 5,
            "8": 6,
            "9": 7,
            "10": 8,
            "J": 9,
            "Q": 10,
            "K": 11,
            "A": 12
        }

    def 生成全部扑克牌(self):
        全部扑克牌数字 = torch.arange(13, dtype=self.dtype, device=self.device)
        全部扑克牌数字 = 全部扑克牌数字.tile(4*self.扑克牌数)
        全部扑克牌花色 = torch.arange(4, dtype=self.dtype, device=self.device)
        全部扑克牌花色 = 全部扑克牌花色.repeat_interleave(13*self.扑克牌数)
        return 全部扑克牌数字, 全部扑克牌花色

    def 删除已有的牌(self, 全部扑克牌数字, 全部扑克牌花色, 删除的牌):
        全部扑克牌数字 = 全部扑克牌数字.tolist()
        全部扑克牌花色 = 全部扑克牌花色.tolist()
        删除的牌ID = []
        for 花色, 数字 in 删除的牌:
            删除的牌ID.append(self.花色字典[花色] * 13 + self.数字字典[数字])
        全部扑克牌数字 = torch.tensor(
            [全部扑克牌数字[i] for i in range(len(全部扑克牌数字)) if i not in 删除的牌ID],
            dtype=self.dtype, device=self.device
        )
        全部扑克牌花色 = torch.tensor(
            [全部扑克牌花色[i] for i in range(len(全部扑克牌花色)) if i not in 删除的牌ID],
            dtype=self.dtype, device=self.device
        )
        return 全部扑克牌数字, 全部扑克牌花色

    def 解析已有的牌(self, 游戏轮数, 已有的牌=[]):
        已有的扑克牌数字, 已有的扑克牌花色 = [], []
        for 花色, 数字 in 已有的牌:
            已有的扑克牌数字.append(self.数字字典[数字])
            已有的扑克牌花色.append(self.花色字典[花色])
        已有的扑克牌数字 = torch.tensor(已有的扑克牌数字, dtype=self.dtype, device=self.device)
        已有的扑克牌花色 = torch.tensor(已有的扑克牌花色, dtype=self.dtype, device=self.device)
        已有的扑克牌数字 = 已有的扑克牌数字.repeat(游戏轮数).view(游戏轮数, -1)
        已有的扑克牌花色 = 已有的扑克牌花色.repeat(游戏轮数).view(游戏轮数, -1)
        return 已有的扑克牌数字, 已有的扑克牌花色

    def 发牌(self, 游戏轮数, 已有的牌=[], 不能选的牌=[]):
        全部扑克牌数字, 全部扑克牌花色 = self.生成全部扑克牌()
        全部扑克牌数字, 全部扑克牌花色 = self.删除已有的牌(全部扑克牌数字, 全部扑克牌花色, 已有的牌 + 不能选的牌)
        扑克牌ID = torch.ones((游戏轮数, len(全部扑克牌数字)), device=self.device)
        扑克牌ID = torch.multinomial(扑克牌ID, 7 - len(已有的牌))
        if len(已有的牌) > 0:
            已有的扑克牌数字, 已有的扑克牌花色 = self.解析已有的牌(游戏轮数, 已有的牌)
            扑克牌数字 = torch.concat([已有的扑克牌数字, 全部扑克牌数字[扑克牌ID]], dim=1)
            扑克牌花色 = torch.concat([已有的扑克牌花色, 全部扑克牌花色[扑克牌ID]], dim=1)
        else:
            扑克牌数字 = 全部扑克牌数字[扑克牌ID]
            扑克牌花色 = 全部扑克牌花色[扑克牌ID]
        return 扑克牌数字, 扑克牌花色

    def 扑克牌计数(self, 扑克牌数字, 扑克牌花色):
        游戏轮数, 扑克牌数 = 扑克牌数字.shape
        计数矩阵 = torch.zeros((游戏轮数, 13, 4), dtype=self.dtype, device=self.device)
        for i in range(扑克牌数):
            计数矩阵[torch.arange(游戏轮数), 扑克牌数字[:, i], 扑克牌花色[:, i]] += 1
        return 计数矩阵

    def 皇家同花顺(self, 计数矩阵):
        kernel = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.float16, device=self.device).reshape(1, 13, 1)
        result = torch.nn.functional.conv1d((计数矩阵 > 0).to(torch.float16), kernel)
        result = (result > 5 - self.eps).view(计数矩阵.shape[0], -1).any(dim=-1)
        return result

    def 同花顺不含A2345(self, 计数矩阵):
        计数矩阵 = 计数矩阵.unsqueeze(1)
        kernel = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float16, device=self.device).reshape(1, 1, 5, 1)
        result = torch.nn.functional.conv2d((计数矩阵 > 0).to(torch.float16), kernel)
        result = (result > 5 - self.eps).view(计数矩阵.shape[0], -1).any(dim=-1)
        return result

    def 同花顺A2345(self, 计数矩阵):
        kernel = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float16, device=self.device).reshape(1, 13, 1)
        result = torch.nn.functional.conv1d((计数矩阵 > 0).to(torch.float16), kernel)
        result = (result > 5 - self.eps).view(计数矩阵.shape[0], -1).any(dim=-1)
        return result

    def 同花顺(self, 计数矩阵):
        牌型 = ["同花顺:不含A2345", "同花顺:A2345"]
        result = [self.同花顺不含A2345(计数矩阵), self.同花顺A2345(计数矩阵)]
        return 牌型, result

    def 四条(self, 计数矩阵):
        result = (计数矩阵.sum(dim=-1) >= 4).any(dim=-1)
        return result

    def 葫芦(self, 计数矩阵):
        计数矩阵 = 计数矩阵.sum(dim=-1)
        三张, 两张 = 计数矩阵 >= 3, 计数矩阵 >= 2
        case_1 = 三张.sum(dim=-1) >= 2
        case_2 = torch.logical_and(torch.logical_and(torch.logical_not(三张), 两张).sum(dim=-1) >= 1, 三张.sum(dim=-1) >= 1)
        result = torch.logical_or(case_1, case_2)
        return result

    def 同花(self, 计数矩阵):
        result = (计数矩阵.sum(dim=-2) >= 5).any(dim=-1)
        return result

    def 顺子不含A2345(self, 计数矩阵):
        计数矩阵 = (计数矩阵.sum(dim=-1) > 0).unsqueeze(1)
        kernel = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float16, device=self.device).reshape(1, 1, 5)
        result = torch.nn.functional.conv1d((计数矩阵 > 0).to(torch.float16), kernel)
        result = (result > 5 - self.eps).view(计数矩阵.shape[0], -1)
        数字字典 = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        牌型, result_ = [], []
        for i in range(13 - 5, -1, -1):
            牌型.append("".join(["顺子:"] + [数字字典[j] for j in range(i, i + 5)]))
            result_.append(result[:, i])
        return 牌型, result_

    def 顺子A2345(self, 计数矩阵):
        计数矩阵 = (计数矩阵.sum(dim=-1) > 0).unsqueeze(1)
        kernel = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float16, device=self.device).reshape(1, 1, 13)
        result = torch.nn.functional.conv1d((计数矩阵 > 0).to(torch.float16), kernel)
        result = (result > 5 - self.eps).view(计数矩阵.shape[0], -1).any(dim=-1)
        return result

    def 顺子(self, 计数矩阵):
        牌型, result = self.顺子不含A2345(计数矩阵)
        牌型 = 牌型 + ["顺子:A2345"]
        result = result + [self.顺子A2345(计数矩阵)]
        return 牌型, result

    def 三条(self, 计数矩阵):
        条子 = 计数矩阵.sum(dim=-1) >= 3
        rank = torch.zeros(计数矩阵.shape[0], dtype=self.dtype, device=self.device)
        for i in range(12, -1, -1):
            rank = rank * (1 + 13 * 条子[:, i]) + (i + 1) * 条子[:, i]
        while (rank > 13).sum() > 0:
            rank[rank > 13] //= 14
        数字字典 = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        牌型, result = [], []
        for num in range(12, -1, -1):
            牌型.append(f"三条:三{数字字典[num]}")
            result.append(rank == num + 1)
        return 牌型, result

    def 两对(self, 计数矩阵):
        result = (计数矩阵.sum(dim=-1) >= 2).sum(dim=-1) >= 2
        return result

    def 一对(self, 计数矩阵):
        result = (计数矩阵.sum(dim=-1) >= 2).sum(dim=-1) >= 1
        return result

    def 两对和一对(self, 计数矩阵):
        对子 = 计数矩阵.sum(dim=-1) >= 2
        rank = torch.zeros(计数矩阵.shape[0], dtype=self.dtype, device=self.device)
        for i in range(12, -1, -1):
            rank = rank * (1 + 13 * 对子[:, i]) + (i + 1) * 对子[:, i]
        while (rank > 13 * 14 + 13).sum() > 0:
            rank[rank > 13 * 14 + 13] //= 14
        数字字典 = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        牌型, result = [], []
        for num1 in range(12, -1, -1):
            for num2 in range(num1 - 1, -1, -1):
                牌型.append(f"两对:对{数字字典[num1]}和对{数字字典[num2]}")
                result.append(rank == (num1 + 1) * 14 + (num2 + 1))
        for num1 in range(12, -1, -1):
            牌型.append(f"一对:对{数字字典[num1]}")
            result.append(rank == num1 + 1)
        return 牌型, result

    def 高牌(self, 扑克牌数字):
        最大数字 = 扑克牌数字.max(dim=-1).values
        数字字典 = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        牌型, result = [], []
        for num in range(12, 6, -1): # 不存在 <= 8 的高牌
            牌型.append(f"高牌:{数字字典[num]}")
            result.append(最大数字 == num)
        return 牌型, result

    def 更新统计值(self, 牌型, 统计值, 标记, result):
        统计值[牌型] = 统计值.get(牌型, 0) + torch.logical_and(标记, result).sum(dim=-1).cpu().tolist()
        标记 = torch.logical_and(标记, torch.logical_not(result))
        return 统计值, 标记
        
    def 计算牌型(self, 扑克牌数字, 扑克牌花色):
        计数矩阵 = self.扑克牌计数(扑克牌数字, 扑克牌花色)
        checkers = [
            self.皇家同花顺,
            self.同花顺,
            self.四条,
            self.葫芦,
            self.同花,
            self.顺子,
            self.三条,
            self.两对和一对,
        ]
        统计值 = {}
        标记 = torch.ones(计数矩阵.shape[0], dtype=torch.bool, device=self.device)
        for checker in checkers:
            检测结果 = checker(计数矩阵)
            if isinstance(检测结果, tuple):
                for 牌型, result in zip(*检测结果):
                    统计值, 标记 = self.更新统计值(牌型, 统计值, 标记, result)
            else:
                result = 检测结果
                统计值, 标记 = self.更新统计值(checker.__name__, 统计值, 标记, result)
        检测结果 = self.高牌(扑克牌数字)
        for 牌型, result in zip(*检测结果):
            统计值, 标记 = self.更新统计值(牌型, 统计值, 标记, result)
        return 统计值

    def 计算概率(self, 统计值):
        num_round = sum([统计值[i] for i in 统计值])
        概率 = {i: 统计值[i] / num_round for i in 统计值}
        return 概率

    def __call__(self, 迭代次数=1, batch_size=2**20, 已有的牌=[]):
        统计值 = {}
        for iter in tqdm(range(迭代次数)):
            扑克牌数字, 扑克牌花色 = self.发牌(batch_size, 已有的牌=已有的牌)
            result = self.计算牌型(扑克牌数字, 扑克牌花色)
            for 牌型 in result:
                统计值[牌型] = 统计值.get(牌型, 0) + result[牌型]
        return self.计算概率(统计值)

    def 计算胜率(self, 己方概率, 对方概率):
        p1 = [己方概率[i] for i in 己方概率]
        p2 = [对方概率[i] for i in 对方概率]
        win = []
        for i in range(len(p1)):
            p = p1[i] * (sum(p2[i+1:], 0) ** (self.人数 - 1))
            win.append(p)
        loss = []
        for i in range(len(p1)):
            p = p1[i] * (1 - (1 - sum(p2[:i], 0)) ** (self.人数 - 1))
            loss.append(p)
        win = sum(win)
        loss = sum(loss)
        tie = 1 - win - loss
        return win, tie, loss

    def show(self, 概率):
        message = ""
        last_prefix = ""
        for 牌型 in 概率:
            if ":" in 牌型:
                prefix = 牌型[:牌型.index(":")]
            else:
                prefix = 牌型
            if prefix != last_prefix:
                p_win = sum([概率[i] for i in 概率 if i.startswith(prefix)])
                message += f"- {prefix}: %.5f" % (p_win * 100) + "%\n"
            if ":" in 牌型:
                message += f"    - {牌型[len(prefix) + 1:]}: %.5f" % (概率[牌型] * 100) + "%\n"
            last_prefix = prefix
        return message


def solve(device, 人数, 扑克牌数, num_iterations, batch_size, colors, numbers):
    手上的牌, 场上的牌 = [], []
    for i, color, number in zip(range(len(colors)), colors, numbers):
        if color == "?" or number == "?":
            continue
        if i < 2:
            手上的牌.append((color, number))
        else:
            场上的牌.append((color, number))
    solver = 德州扑克求解器(人数, 扑克牌数, device=device)
    p_our = solver(num_iterations, batch_size, 已有的牌 = 手上的牌 + 场上的牌)
    p_other = solver(num_iterations, batch_size, 已有的牌 = 场上的牌)
    win, tie, loss = solver.计算胜率(p_our, p_other)
    return win, tie, loss, p_our, p_other


with st.sidebar:
    with st.expander("Config", expanded=True):
        total_players = st.number_input("Total Players", min_value=2, max_value=100, step=1, value=2)
        total_decks_of_cards = st.number_input("Total Decks of Cards", min_value=1, max_value=100, step=1, value=1)
        num_iterations = st.number_input("Total Iterations", min_value=1, max_value=100, step=1, value=1)
        batch_size = st.number_input("Batch Size (2^n)", min_value=1, max_value=32, step=1, value=20)
        device = st.selectbox("Device", ["cuda", "mps", "cpu"], index=0))
    colors, numbers = [], []
    with st.expander("Cards in your hand", expanded=True):
        for i in [1, 2]:
            column_color, column_number = st.columns(2)
            with column_color:
                colors.append(st.selectbox(f"Decor {i}", ["?", "黑桃", "红桃", "梅花", "方片"], index=0))
            with column_number:
                numbers.append(st.selectbox(f"Number {i}", ["?", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"], index=0))
    with st.expander("Cards on the desk", expanded=True):
        for i in [3, 4, 5, 6, 7]:
            column_color, column_number = st.columns(2)
            with column_color:
                colors.append(st.selectbox(f"Decor {i}", ["?", "黑桃", "红桃", "梅花", "方片"], index=0))
            with column_number:
                numbers.append(st.selectbox(f"Number {i}", ["?", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"], index=0))
run = st.button("Run", type="primary")
if run:
    win, tie, loss, p_our, p_other = solve(device, total_players, total_decks_of_cards, num_iterations, 2**batch_size, colors, numbers)
    
    st.markdown("### Results")
    with st.expander("Results", expanded=True):
        column_win, column_tie, column_loss = st.columns(3)
        with column_win:
            st.slider("Win", min_value=0.0, max_value=100.0, step=0.0001, value=win * 100)
        with column_tie:
            st.slider("Tie", min_value=0.0, max_value=100.0, step=0.0001, value=tie * 100)
        with column_loss:
            st.slider("Loss", min_value=0.0, max_value=100.0, step=0.0001, value=loss * 100)

    st.markdown("### Details")
    data = pd.DataFrame()
    data["Category"] = [i for i in p_our]
    data["Probability (our)"] = ["%.4f" % (p_our[i] * 100) + "%" for i in p_our]
    data["Probability (other players)"] = ["%.4f" % (p_other[i] * 100) + "%" for i in p_other]
    st.data_editor(data, use_container_width=True)

    st.markdown("### Chart")
    p = [p_our[i] * 100 for i in p_our]
    p = [sum(p[:i+1], 0) for i in range(len(p))]
    data["Probability Prefix Sum (our)"] = p
    p = [p_other[i] * 100 for i in p_other]
    p = [-sum(p[:i+1], 0) for i in range(len(p))]
    data["Probability Prefix Sum (other players)"] = p
    st.area_chart(data, y=["Probability Prefix Sum (our)", "Probability Prefix Sum (other players)"])
