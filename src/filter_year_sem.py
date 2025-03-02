import json
import pandas as pd


def filter_course_by_year_and_sem(year, sem):
    # Load course info
    with open(
        "/home/gokul/gigs/hacktheburgh/data/course_desc/informatics_course_info.json",
        "r",
    ) as file:
        course_info = json.load(file)

    # Course ID list
    course_id_list = [
        course["course_code"]
        for course in course_info
        if course["semester"] == f"Semester {sem}"
    ]

    course_df = pd.read_parquet(
        "/home/gokul/gigs/hacktheburgh/data/uoe/course_details.parquet",
    )

    course_year_filtered = course_df[course_df["Programme Year"] == year][
        "Course Code"
    ].tolist()

    course_id_list = list(set(course_id_list).intersection(set(course_year_filtered)))

    return course_id_list


if __name__ == "__main__":
    print(filter_course_by_year_and_sem(1, 2))
