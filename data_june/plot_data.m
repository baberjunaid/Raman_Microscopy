d1  = 'control_Nucleous/';
d2 = 'MV_DATA/';
d3 = 'SARS_Nucleous/';

dd = dir([d1 '*.txt']);

for i=1:length(dd)
    t = readtable([d1 dd(i).name]);
    t = table2array(t(:,1:2));
    plot(t(:,1), t(:,2)), hold on;
end

dd = dir([d2 '*.txt']);
figure
for i=1:length(dd)
    t = readtable([d2 dd(i).name]);
    t = table2array(t(:,1:2));
    plot(t(:,1), t(:,2)), hold on;
end

dd = dir([d3 '*.txt']);
figure
for i=1:length(dd)
    t = readtable([d3 dd(i).name]);
    t = table2array(t(:,1:2));
    plot(t(:,1), t(:,2)), hold on;
end


