%function:基于因子分析的部分IT公司成长性评价研究 
%
%
%
%第一步：因子分析
%
[X,textdata] = xlsread('/ITdata.xls');
X = X(:,1:end)    % 提取X的第3至最后一列，即要分析的数据  %这里读取数据需要格外注意
varname = textdata(1,3:end)  %提取textdata的第1行，第3至最后一列，即变量名
obsname = textdata(2:end,1)%提取textdata的第1列，第2行至最后一行，即公司名
[lambda,psi,T,stats,F] = factoran(X,3);% 进行因子旋转(最大方差旋转法)
[varname' num2cell(lambda)] 
[varname' num2cell(psi)] 
Contribut = 100*sum(lambda.^2)/7   %计算贡献率，因子载荷阵的列元素之和除以维数
CumCont = cumsum(Contribut)    %计算累积贡献率
%
%第二步：计算因子得分并排序
%
obsF = [obsname, num2cell(F)]; %将公司名与因子得分放在一个元胞数组中显示
F1 = sortrows(obsF, -2);    % 按盈利能力得分排序
F2 = sortrows(obsF, -3);   % 按运营能力得分排序
F3 = sortrows(obsF, -4);   % 按发展前景得分排序
head = {'公司','盈利能力','运营能力','发展前景'};
result1 = [head; F1]
result2 = [head; F2]
result3 = [head; F3]

%
%第三步：对显示结果进行优化
%
scatter3(F(:,1),F(:,2),F(:,3),'k'); 
xlabel('盈利能力');
ylabel('运营能力');
zlabel('发展前景');
text(F(:,1)+0.03,F(:,2),F(:,3), obsname, 'fontsize', 5);%添加各散点的标注
