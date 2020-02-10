import datetime
from collections import defaultdict
data = [
{'closing_price': 102.06,
'date': datetime.datetime(2014, 8, 29, 0, 0),
'symbol': 'AAPL'},
# ...
]

max_aapl_price = max(row["closing_price"]
                    for row in data
                    if row["symbol"] == "AAPL")

print(max_aapl_price)

# group rows by symbol
by_symbol = defaultdict(list)
for row in data:
    by_symbol[row["symbol"]].append(row)


# use a dict comprehension to find the max for each symbol
max_price_by_symbol = { symbol : max(row["closing_price"]
                        for row in grouped_rows)
                        for symbol, grouped_rows in by_symbol.items() }
print(max_price_by_symbol)

def picker(field_name):
    """returns a function that picks a field out of a dict"""
    return lambda row: row[field_name]


def pluck(field_name, rows):
    """turn a list of dicts into the list of field_name values"""
    return map(picker(field_name), rows)


def group_by(grouper, rows, value_transform=None):
    # key is output of grouper, value is list of rows
    grouped = defaultdict(list)
    for row in rows:
        grouped[grouper(row)].append(row)
    if value_transform is None:
        return grouped
    else:
        return { key : value_transform(rows)
                for key, rows in grouped.items() }


max_price_by_symbol = group_by(picker("symbol"),
                                      data,
                                      lambda rows: max(pluck("closing_price", rows)))
print(max_price_by_symbol)


def percent_price_change(yesterday, today):
    return today["closing_price"] / yesterday["closing_price"] - 1

def day_over_day_changes(grouped_rows):
    # sort the rows by date
    ordered = sorted(grouped_rows, key=picker("date"))
    # zip with an offset to get pairs of consecutive days
    return [{ "symbol" : today["symbol"],
            "date" : today["date"],
            "change" : percent_price_change(yesterday, today) }
            for yesterday, today in zip(ordered, ordered[1:])]


# key is symbol, value is list of "change" dicts
changes_by_symbol = group_by(picker("symbol"), data, day_over_day_changes)

# collect all "change" dicts into one big list
all_changes = [change
                for changes in changes_by_symbol.values()
                for change in changes]

#max(all_changes, key=picker("change"))
#min(all_changes, key=picker("change"))


def combine_pct_changes(pct_change1, pct_change2):
    return (1 + pct_change1) * (1 + pct_change2) - 1

def overall_change(changes):
    return reduce(combine_pct_changes, pluck("change", changes))

overall_change_by_month = group_by(lambda row: row['date'].month,
                                    all_changes,
                                    overall_change)