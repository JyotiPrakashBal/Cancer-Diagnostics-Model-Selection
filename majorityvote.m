

for i= 1:samples
in = models(i,:);
[count,values]=hist(in,unique(in));
    
%Find the array index corresponding to  the value with the most occurrences
[Vmax,argmax]=max(count);
%Output function with most occurrences
 out(i)=values(argmax);

end
out=out';

correct=sum(out==test_Labels);                                                                                                        
acc=100*correct/length(test_Labels);
disp(acc);


      [~,~,T,auc] = perfcurve(testLabels,out,2);
  