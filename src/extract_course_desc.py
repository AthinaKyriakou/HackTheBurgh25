import json
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import time

# Load the HTML file
# file_path = "/home/gokul/gigs/hacktheburgh/data/course_desc/cg_page_src.txt"
# with open(file_path, "r", encoding="utf-8") as file:
# html_content = file.read()

base_url = "http://www.drps.ed.ac.uk/24-25/dpt/"

# URL of the department course list page
department_url = urljoin(base_url, "cx_sb_infr.htm")


def extract_course_info(url):

    response = requests.get(url)

    # Check if request was successful
    if response.status_code != 200:
        print(f"Failed to fetch the URL: {url}")
        return

    # Parse HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract course title & code
    course_code = soup.find("meta", {"name": "modcode"})["content"]
    course_title = soup.find("meta", {"name": "modname"})["content"]

    # Extract key details from tables
    def extract_table_data(table_caption):
        """Extracts structured data from tables based on their caption."""
        table = soup.find("caption", string=table_caption)
        if not table:
            return None

        extracted_data = {}
        rows = table.find_parent("table").find_all("tr")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                key = cols[0].text.strip().replace(":", "")
                value = cols[1].text.strip().replace("\n", " ")
                extracted_data[key] = value

        return extracted_data

    # Extract different sections dynamically
    # course outline table
    course_outline = extract_table_data("Course Outline")
    course_credits = int(course_outline.get("SCQF Credits", "-99"))
    course_summary = course_outline.get("Summary", "")
    course_desc = course_outline.get("Course description", "")

    # entry requirements table
    entry_requirements = extract_table_data(
        "Entry Requirements (not applicable to Visiting Students)"
    )
    prerequisites = entry_requirements.get("Pre-requisites", "")

    # Initialize empty lists for mandatory and recommended courses
    mandatory_courses = []
    recommended_courses = []

    if prerequisites:
        # Remove the unnecessary prefix text
        prerequisites = prerequisites.replace("Students MUST have passed:", "").strip()
        prerequisites = prerequisites.replace(
            "It is RECOMMENDED that students have passed", "|"
        ).strip()

        # Split mandatory and recommended parts
        parts = prerequisites.split("|")

        # Extract mandatory courses (first part)
        if len(parts) > 0 and parts[0].strip():
            mandatory_courses = [course.strip() for course in parts[0].split(" AND ")]

        # Extract recommended courses (second part, if available)
        if len(parts) > 1 and parts[1].strip():
            recommended_courses = [course.strip() for course in parts[1].split(" AND ")]

    prohibited_combinations = entry_requirements.get("Prohibited Combinations", "")
    other_requirements = entry_requirements.get("Other Requirements", "")

    # Clean "Prohibited Combinations" to be a list
    if prohibited_combinations:
        prohibited_combinations = prohibited_combinations.replace(
            "Students MUST NOT also be taking", ""
        ).strip()
        prohibited_combinations = [
            course.strip() for course in prohibited_combinations.split("AND")
        ]

    # course delivery table
    delivery_info = extract_table_data("Course Delivery Information")
    # print(delivery_info)
    semester = delivery_info.get("Course Start", "")
    assessment = delivery_info.get("Assessment (Further Info)", "").strip()

    # learning outcomes table
    learning_outcomes = soup.find("caption", string="Learning Outcomes")
    if learning_outcomes:
        learning_outcomes = learning_outcomes.find_parent("table").find_all("li")
        learning_outcomes = [lo.text.strip() for lo in learning_outcomes]
        learning_outcomes = "\n".join(learning_outcomes)
    else:
        learning_outcomes = ""

    # final JSON
    return {
        "course_code": course_code,
        "course_title": course_title,
        "course_summary": course_summary,
        "course_desc": course_desc,
        "course_credits": course_credits,
        "semester": semester,
        "assessment": assessment,
        "prerequisites": {
            "mandatory": mandatory_courses,
            "recommended": recommended_courses,
        },
        "prohibited_combinations": prohibited_combinations,
        "other_requirements": other_requirements,
        "learning_outcomes": learning_outcomes,
    }


def get_course_links(department_url):
    """Extracts all course links from the department page."""
    response = requests.get(department_url)
    if response.status_code != 200:
        print(f"Failed to fetch department page: {department_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract course links
    course_links = []
    for link in soup.find_all("a", href=True):
        if link["href"].startswith("cxinfr"):  # Course pages start with 'cxinfr'
            full_url = urljoin(base_url, link["href"])
            course_links.append(full_url)

    return course_links


if __name__ == "__main__":

    course_links = get_course_links(department_url)

    # Iterate through each course link and extract information
    all_courses_info = []
    for idx, course_url in enumerate(course_links):
        print(f"Processing {idx+1}/{len(course_links)}: {course_url}")
        course_info = extract_course_info(course_url)
        all_courses_info.append(course_info)
        time.sleep(1)  # Pause between requests to avoid overload

    # Save results as JSON
    output_json_path = (
        "/home/gokul/gigs/hacktheburgh/data/course_desc/informatics_course_info.json"
    )
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(all_courses_info, json_file, indent=4)

    print(f"All course information saved to: {output_json_path}")
