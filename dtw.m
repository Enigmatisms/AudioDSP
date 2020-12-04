function dist = dtw(t,r)
%%%%%�����������%%%%%%%%
arr_1 = size(t,1);
arr_2 = size(r,1);

d = zeros(arr_1,arr_2);% ֡ƥ��������

for i = 1:arr_1
    for j = 1:arr_2
        d(i,j) = sum((t(i,:)-r(j,:)).^2);
    end
end

%%%%%%%%%%%%%%%%%�ۻ��������%%%%%%%%%%%%%%%%%%%
Distance =  ones(arr_1,arr_2) * realmax;
Distance(1,1) = d(1,1);

% ��̬�滮
for i = 2:arr_1
    for j = 1:arr_2
        Distance1 = Distance(i-1,j);
	if j>1
		Distance2 = Distance(i-1,j-1);
    else
        Distance2 = realmax;%����������ʵ����
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