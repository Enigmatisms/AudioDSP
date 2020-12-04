function dist = dtw(t,r)
%%%%%这个是主函数%%%%%%%%
arr_1 = size(t,1);
arr_2 = size(r,1);

d = zeros(arr_1,arr_2);% 帧匹配距离矩阵

for i = 1:arr_1
    for j = 1:arr_2
        d(i,j) = sum((t(i,:)-r(j,:)).^2);
    end
end

%%%%%%%%%%%%%%%%%累积距离矩阵%%%%%%%%%%%%%%%%%%%
Distance =  ones(arr_1,arr_2) * realmax;
Distance(1,1) = d(1,1);

% 动态规划
for i = 2:arr_1
    for j = 1:arr_2
        Distance1 = Distance(i-1,j);
	if j>1
		Distance2 = Distance(i-1,j-1);
    else
        Distance2 = realmax;%赋给他最大的实参数
	end

	if j>2
		Distance3 = Distance(i-1,j-2);
    else
        Distance3 = realmax;
	end

	Distance(i,j) = d(i,j) + min([Distance1,Distance2,Distance3]);
    end
end

dist = Distance(arr_1,arr_2);