from collections import defaultdict
class Table:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def __repr__(self):
        """pretty representation of the table: columns then rows"""
        return str(self.columns) + "\n" + "\n".join(map(str, self.rows))

    def insert(self, row_values):
        if len(row_values) != len(self.columns):
            raise TypeError("wrong number of elements")
        row_dict = dict(zip(self.columns, row_values))
        self.rows.append(row_dict)

    def update(self, updates, predicate):
        for row in self.rows:
            if predicate(row):
                for column, new_value in updates.items():
                    row[column] = new_value

    def delete(self, predicate=lambda row: True):
        """delete all rows matching predicate
        or all rows if no predicate supplied"""
        self.rows = [row for row in self.rows if not(predicate(row))]


    def select(self, keep_columns=None, additional_columns=None):
        if keep_columns is None: # if no columns specified,
            keep_columns = self.columns # return all columns
        if additional_columns is None:
            additional_columns = {}
        # new table for results
        result_table = Table(keep_columns + list(additional_columns.keys()))
        for row in self.rows:
            new_row = [row[column] for column in keep_columns]
            for column_name, calculation in additional_columns.items():
                new_row.append(calculation(row))
            result_table.insert(new_row)
        return result_table


    def where(self, predicate=lambda row: True):
        """return only the rows that satisfy the supplied predicate"""
        where_table = Table(self.columns)
        where_table.rows = filter(predicate, self.rows)
        return where_table

    def limit(self, num_rows):
        """return only the first num_rows rows"""
        limit_table = Table(self.columns)
        limit_table.rows = self.rows[:num_rows]
        return limit_table

    def group_by(self, group_by_columns, aggregates, having=None):
        grouped_rows = defaultdict(list)
        # populate groups
        for row in self.rows:
            key = tuple(row[column] for column in group_by_columns)
            grouped_rows[key].append(row)
        # result table consists of group_by columns and aggregates
        result_table = Table(group_by_columns + list(aggregates.keys()))
        for key, rows in grouped_rows.items():
            if having is None or having(rows):
                new_row = list(key)
                for aggregate_name, aggregate_fn in aggregates.items():
                    new_row.append(aggregate_fn(rows))
                result_table.insert(new_row)
        return result_table


    def order_by(self, order):
        new_table = self.select() # make a copy
        new_table.rows.sort(key=order)
        return new_table