%data = readtable ("clean_data.csv");
%disp(data)
%columns_with_missing = data.Properties.VariableNames(any(ismissing(data), 1));
column_names = data.Properties.VariableNames;
missing_counts = sum(ismissing(data));

for i = 1:length(column_names)
    if missing_counts(i) > 0
        fprintf('Column Name: %s, Total NA Values: %d\n', column_names{i}, missing_counts(i));
    end
end