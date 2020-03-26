import BetweennessCentrality as bet

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
                (2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
                (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

for user in bet.users:
    user["endorses"] = [] # add one list to track outgoing endorsements
    user["endorsed_by"] = [] # and another to track endorsements


for source_id, target_id in endorsements:
    bet.users[source_id]["endorses"].append(bet.users[target_id])
    bet.users[target_id]["endorsed_by"].append(bet.users[source_id])


endorsements_by_id = [(user["id"], len(user["endorsed_by"]))
                       for user in bet.users]

sorted(endorsements_by_id,
       key=lambda pair: pair[1],
       reverse=True)

print("---------------------------------------")
print(endorsements_by_id)


def page_rank(users, damping = 0.85, num_iters = 100):
    # initially distribute PageRank evenly
    num_users = len(users)
    pr = { user["id"] : 1 / num_users for user in users }
    # this is the small fraction of PageRank
    # that each node gets each iteration
    base_pr = (1 - damping) / num_users

    for __ in range(num_iters):
        next_pr = { user["id"] : base_pr for user in users }
        for user in users:
            # distribute PageRank to outgoing links
            links_pr = pr[user["id"]] * damping

            for endorsee in user["endorses"]:
                next_pr[endorsee["id"]] += links_pr / len(user["endorses"])

        pr = next_pr

    return pr


print(page_rank(bet.users))