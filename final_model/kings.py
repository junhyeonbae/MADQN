import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch



def king_adj2(n) :

    A = np.zeros((n**2, n**2))

    for i in range(n**2):

        if i // n == 0  : #첫번째 행

            if i % n  == 0 : #첫번째 열 : 몫이 0 이어야 함

                A[i, i+1] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

            elif i % n == n-1 : #마지막 열  몫이 n-1 이어야 함

                A[i, i-1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1

            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

        elif i // n == n-1 : #마지막 행

            if i % n ==  0 : #첫번째 열

                A[i, i-n] = 1
                A[i, i+1-n] = 1
                A[i, i+1] = 1
            elif i % n  == n-1 : #마지막 열

                A[i, i-n] = 1
                A[i, i-1-n] = 1
                A[i, i-1] = 1
            else: #중간열
                A[i, i - 1] = 1
                A[i, i + 1] = 1
                A[i, i - 1-n] = 1
                A[i, i-n] = 1
                A[i, i+ 1-n] = 1

        else:  #중간 행에 포함되는 것



                if i % n == 0 : #첫번째 열
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i+1] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1

                elif i % n == n-1 : #마지막 열
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i-1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                else:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i-1] = 1
                    A[i, i+1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1


    for i in range(n**2):
        A[i,i] = 1

    return A


def overlap_area(a, b, x, y, side_length):
    # 각 정사각형의 좌표를 계산 (왼쪽, 오른쪽, 위, 아래)
    square1 = {'left': a - side_length / 2, 'right': a + side_length / 2,
               'top': b + side_length / 2, 'bottom': b - side_length / 2}
    square2 = {'left': x - side_length / 2, 'right': x + side_length / 2,
               'top': y + side_length / 2, 'bottom': y - side_length / 2}

    # 겹치는 영역의 가로와 세로 길이를 계산
    overlap_width = min(square1['right'], square2['right']) - max(square1['left'], square2['left'])
    overlap_height = min(square1['top'], square2['top']) - max(square1['bottom'], square2['bottom'])

    # 겹치는 영역이 있다면 넓이를 계산, 없다면 0을 반환
    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


# 중앙의 위치가 (a, b)와 (x, y)이고, 한 변의 길이가 10인 두 정사각형의 겹치는 영역의 넓이를 계산
a, b = 5, 5  # 첫 번째 정사각형의 중앙 좌표
x, y = 7, 7  # 두 번째 정사각형의 중앙 좌표
side_length = 10

overlap = overlap_area(a, b, x, y, side_length)
print(f'The overlapping area is: {overlap}')

import numpy as np


def overlap_area(a, b, x, y, side_length1, side_length2):
    # 겹치는 영역의 가로와 세로 길이를 계산합니다
    overlap_width = min(a + side_length1 / 2, x + side_length2 / 2) - max(a - side_length1 / 2, x - side_length2 / 2)
    overlap_height = min(b + side_length1 / 2, y + side_length2 / 2) - max(b - side_length1 / 2, y - side_length2 / 2)

    # 겹치는 영역이 있다면 넓이를 계산, 없다면 0을 반환
    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


# 정사각형들의 중앙 좌표와 한 변의 길이 (상반부는 10, 하반부는 5)
centers = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
side_lengths = [10, 10, 10, 5, 5, 5]

# 겹치는 부분의 넓이를 저장할 행렬 초기화
overlap_matrix = np.zeros((6, 6))

# 각 정사각형 쌍에 대한 겹치는 영역의 넓이 계산
for i in range(6):
    for j in range(i + 1, 6):  # 중복 계산을 피하기 위해 j는 i+1부터 시작
        a, b = centers[i]
        x, y = centers[j]
        overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap  # 대칭성을 이용

print(overlap_matrix)

import numpy as np


def overlap_area(a, b, x, y, side_length1, side_length2):
    overlap_width = min(a + side_length1 / 2, x + side_length2 / 2) - max(a - side_length1 / 2, x - side_length2 / 2)
    overlap_height = min(b + side_length1 / 2, y + side_length2 / 2) - max(b - side_length1 / 2, y - side_length2 / 2)

    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


centers = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
side_lengths = [10, 10, 10, 5, 5, 5]

overlap_matrix = np.zeros((6, 6))
square_areas = np.zeros(6)  # 각 정사각형의 넓이를 저장할 배열

# 각 정사각형의 넓이 계산
for i in range(6):
    square_areas[i] = side_lengths[i] ** 2

# 겹치는 영역의 넓이 계산
for i in range(6):
    for j in range(i + 1, 6):
        print(i,j)
        a, b = centers[i]
        x, y = centers[j]
        overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap

# 각 행을 해당 정사각형의 넓이로 나눔
for i in range(6):
    overlap_matrix[i, :] /= square_areas[i]

print(overlap_matrix)



def overlap_area(a, b, x, y, side_length1, side_length2):
    overlap_width = min(a + side_length1 / 2, x + side_length2 / 2) - max(a - side_length1 / 2, x - side_length2 / 2)
    overlap_height = min(b + side_length1 / 2, y + side_length2 / 2) - max(b - side_length1 / 2, y - side_length2 / 2)

    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0


centers = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
side_lengths = [10, 10, 10, 5, 5, 5]

overlap_matrix = np.zeros((6, 6))
square_areas = np.zeros(6)  # 각 정사각형의 넓이를 저장할 배열

# 각 정사각형의 넓이 계산
for i in range(6):
    square_areas[i] = side_lengths[i] ** 2

# 겹치는 영역의 넓이 계산
for i in range(6):
    for j in range(i + 1, 6):
        print(i,j)
        a, b = centers[i]
        x, y = centers[j]
        overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap



# 각 행을 해당 정사각형의 넓이로 나눔
for i in range(6):
    overlap_matrix[i, :] /= square_areas[i]

print(overlap_matrix)

import numpy as np


def overlap_area(a, b, x, y, side_length1, side_length2):
    overlap_width = min(a + side_length1 / 2, x + side_length2 / 2) - max(a - side_length1 / 2, x - side_length2 / 2)
    overlap_height = min(b + side_length1 / 2, y + side_length2 / 2) - max(b - side_length1 / 2, y - side_length2 / 2)

    if overlap_width > 0 and overlap_height > 0:
        return overlap_width * overlap_height
    else:
        return 0



#이걸 싹다 함수로 만들수도 있을 것 같은데?
# 이건 pos 값으로 주어지는 것이다.
centers = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
# 이건 args 으로써 주어지는 것이다.
side_lengths = [10, 10, 10, 5, 5, 5]

overlap_matrix = np.zeros((6, 6))
square_areas = np.zeros(6)  # 각 정사각형의 넓이를 저장할 배열

# 각 정사각형의 넓이 계산
for i in range(6):
    square_areas[i] = side_lengths[i] ** 2

# 겹치는 영역의 넓이 계산
for i in range(6):
    for j in range(i + 1, 6):
        a, b = centers[i]
        x, y = centers[j]
        overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
        overlap_matrix[i, j] = overlap
        overlap_matrix[j, i] = overlap

# 각 행을 해당 정사각형의 넓이로 나눔
for i in range(6):
    overlap_matrix[i, :] /= square_areas[i]

print(overlap_matrix)

# 결과 행렬 초기화
result_matrix = np.zeros((6, 2))

result_matrix[:, 0] = np.sum(overlap_matrix[:, :3], axis=1)  # 첫 3개 정사각형에 대한 합
result_matrix[:, 1] = np.sum(overlap_matrix[:, 3:], axis=1)  # 나머지 3개 정사각형에 대한 합

print(result_matrix)


# 2차원 텐서 (예: 행렬)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# 각 행에 대한 L2 norm 계산
row_norms = torch.norm(x, p=2, dim=1)
print(row_norms)

# 각 열에 대한 L2 norm 계산
col_norms = torch.norm(x, p=2, dim=0)
print(col_norms)

for agent_idx in range(3, 6):
    print(agent_idx)

centers = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
centers2 = [(5, 2), (2, 4), (7, 3), (1, 4), (5, 2), (9, 12)]




n_predator1 =3
n_predator2 =3

predator1_view_range = 6
predator2_view_range = 3

def calculate_Overlap_ratio_intake(past, now):

    centers_past = past
    centers_past = [(x + 1, y + 1) for x, y in centers_past]

    centers_now = now
    centers_now = [(x + 1, y + 1) for x, y in centers_now]

    side_lengths = [predator1_view_range*2 for i in range(n_predator1)] + [predator2_view_range*2 for i in range(n_predator2)]

    overlap_matrix = np.zeros((n_predator1+n_predator2, n_predator1+n_predator2))


    # 겹치는 영역의 넓이 계산
    for i in range(n_predator1+n_predator2): #행의 기준이 되는 에이전트를 의미함
        for j in range(n_predator1+n_predator2):
            a, b = centers_now[i]
            x, y = centers_past[j]
            overlap = overlap_area(a, b, x, y, side_lengths[i], side_lengths[j])
            overlap_matrix[i, j] = overlap


    # 결과 행렬 초기화
    result_matrix = np.zeros((n_predator1+n_predator2, 2))

    result_matrix[:, 0] = np.sum(overlap_matrix[:, :n_predator1], axis=1)  # 첫 3개 정사각형에 대한 합
    result_matrix[:, 1] = np.sum(overlap_matrix[:, n_predator2:], axis=1)  # 나머지 3개 정사각형에 대한 합

    return result_matrix


calculate_Overlap_ratio_intake(centers,centers2)


num_set = {3,4,5}
positions = {number: [] for number in num_set}

print(positions)

#
# import matplotlib.pyplot as plt
#
# # 좌표 리스트
# coords = [(1, 2), (2, 3), (3, 4)]
#
# # 좌표 리스트를 x와 y로 분리
# x, y = zip(*coords)
#
# # 산점도 그리기
# plt.scatter(x, y)
#
# # 그래프 제목과 축 이름 설정
# plt.title('Scatter Plot Example')
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
#
# # 그래프 보여주기
# plt.show()
semi_shared = np.ones((10,10))
print(semi_shared)
x_start = y_start =3
x_range = y_range =2
semi_shared[x_start - (x_range - 1):x_start + (x_range + 1), y_start - (y_range - 1): y_start + (y_range + 1)] += 1
print(semi_shared)
