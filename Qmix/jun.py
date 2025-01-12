import numpy as np

a = np.eye(4)
print(a)

b = np.eye(4)[[1,1,1]]
print(b)
#last_onehot_a_n = np.eye(4)[a_n]



N = 3  # 에이전트 수
action_dim = 4  # 각 에이전트가 선택할 수 있는 행동 수

avail_a_n = [
    [1, 0, 1, 1],  # 첫 번째 에이전트의 가능한 행동
    [0, 1, 0, 1],  # 두 번째 에이전트의 가능한 행동
    [1, 1, 1, 0]   # 세 번째 에이전트의 가능한 행동
]

# for avail_a in avail_a_n:
#     print(avail_a)
#     print(np.nonzero(avail_a)[0])
#     print(np.random.choice(np.nonzero(avail_a)[0]))

a_n = [np.random.randint(low= 0, high= 12, dtype=int) for _ in range(10)]
print(a_n)



# a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]