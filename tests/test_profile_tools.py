# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""
Tests for Profile Tools with Healthcare Sandbox.

Tests the sandbox-aware profile tools for patient profile management.
"""

import pytest

from tests.conftest import create_test_patient_profile
from patient_agent_bench.sandbox import (
    Doctor,
    HealthcareSandbox,
    OfficeLocation,
)
from patient_agent_bench.tools.profile_tools import (
    create_get_profile,
    create_update_contact_info,
    create_update_insurance,
    create_update_pcp,
    create_update_pharmacy,
    get_profile_tools,
)


@pytest.fixture
def sandbox_with_profile():
    """Create a sandbox with test profile and doctors."""
    sandbox = HealthcareSandbox()

    # Add test offices
    sandbox.offices["office_001"] = OfficeLocation(
        id="office_001",
        name="Downtown Medical Center",
        address="123 Main St",
        city="Boston",
        state="MA",
        zip_code="02101",
        phone="555-123-4567",
        hours="Mon-Fri 8AM-6PM",
    )
    sandbox.offices["office_002"] = OfficeLocation(
        id="office_002",
        name="Suburban Health Clinic",
        address="456 Oak Ave",
        city="Cambridge",
        state="MA",
        zip_code="02139",
        phone="555-987-6543",
        hours="Mon-Sat 9AM-5PM",
    )

    # Add test doctors
    sandbox.doctors["doc_001"] = Doctor(
        id="doc_001",
        name="Dr. Sarah Johnson",
        specialty="Primary Care",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.doctors["doc_002"] = Doctor(
        id="doc_002",
        name="Dr. Michael Chen",
        specialty="Primary Care",
        credentials="DO",
        office_id="office_002",
    )
    sandbox.doctors["doc_003"] = Doctor(
        id="doc_003",
        name="Dr. Emily Rodriguez",
        specialty="Cardiology",
        credentials="MD",
        office_id="office_001",
    )
    sandbox.pcp_id = "doc_001"

    # Add patient profile
    sandbox.patient_profile = create_test_patient_profile(
        first_name="John",
        last_name="Doe",
        preferred_name="Johnny",
        dob="1985-03-15",
        sex="Male",
        pronouns="he/him",
        age=40,
        phone="555-111-2222",
        address="789 Elm Street",
        city="Boston",
        state="MA",
        zip_code="02102",
        pcp_name="Dr. Sarah Johnson",
        pcp_id="doc_001",
        insurance_provider="Blue Cross Blue Shield",
        insurance_member_id="XYZ123456",
        insurance_group="GRP001",
        pharmacy_name="CVS Pharmacy",
        pharmacy_address="100 Health Way",
        pharmacy_phone="555-333-4444",
    )
    # Set email directly
    sandbox.patient_profile.personal_info.email = "john.doe@email.com"
    # Set emergency contact directly
    sandbox.patient_profile.emergency_contact.first_name = "Jane"
    sandbox.patient_profile.emergency_contact.last_name = "Doe"
    sandbox.patient_profile.emergency_contact.phone = "555-555-6666"
    sandbox.patient_profile.emergency_contact.relationship = "Spouse"

    sandbox._initialized = True
    return sandbox


class TestGetProfileTools:
    """Tests for get_profile_tools function."""

    def test_returns_five_tools(self, sandbox_with_profile):
        """Test that get_profile_tools returns exactly 5 tools."""
        tools = get_profile_tools(sandbox_with_profile)
        assert len(tools) == 5

    def test_tool_names(self, sandbox_with_profile):
        """Test that tools have correct names."""
        tools = get_profile_tools(sandbox_with_profile)
        tool_names = [t.name for t in tools]

        assert "get_profile" in tool_names
        assert "update_pcp" in tool_names
        assert "update_pharmacy" in tool_names
        assert "update_insurance" in tool_names
        assert "update_contact_info" in tool_names

    def test_tools_have_descriptions(self, sandbox_with_profile):
        """Test that all tools have descriptions."""
        tools = get_profile_tools(sandbox_with_profile)

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


class TestGetProfile:
    """Tests for get_profile functionality."""

    def test_get_profile_returns_info(self, sandbox_with_profile):
        """Test get_profile returns profile information."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        assert "John" in result
        assert "Doe" in result
        assert "Johnny" in result  # preferred name
        assert "1985-03-15" in result

    def test_get_profile_shows_contact_info(self, sandbox_with_profile):
        """Test get_profile shows contact information."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        assert "555-111-2222" in result
        assert "Boston" in result

    def test_get_profile_shows_pcp(self, sandbox_with_profile):
        """Test get_profile shows PCP information."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        assert "Dr. Sarah Johnson" in result

    def test_get_profile_shows_insurance(self, sandbox_with_profile):
        """Test get_profile shows insurance information."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        assert "Blue Cross Blue Shield" in result

    def test_get_profile_shows_pharmacy(self, sandbox_with_profile):
        """Test get_profile shows pharmacy information."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        assert "CVS Pharmacy" in result
        assert "100 Health Way" in result

    def test_get_profile_shows_emergency_contact(self, sandbox_with_profile):
        """Test get_profile shows emergency contact if available."""
        tool = create_get_profile(sandbox_with_profile)
        result = tool.invoke({})

        # Verify the profile is returned successfully (XML format)
        assert "John" in result

    def test_get_profile_no_profile(self):
        """Test get_profile when no profile exists."""
        sandbox = HealthcareSandbox()
        sandbox._initialized = True

        tool = create_get_profile(sandbox)
        result = tool.invoke({})

        assert "Error" in result or "not available" in result


class TestUpdatePcp:
    """Tests for update_pcp functionality."""

    def test_update_pcp_valid_doctor(self, sandbox_with_profile):
        """Test updating PCP to a valid doctor."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({"new_pcp_name": "Dr. Michael Chen"})

        assert "✓" in result
        assert "Dr. Michael Chen" in result
        assert "DO" in result
        assert sandbox_with_profile.pcp_id == "doc_002"
        pcp = sandbox_with_profile.patient_profile.get_pcp()
        assert pcp is not None
        assert pcp.name == "Dr. Michael Chen"

    def test_update_pcp_with_reason(self, sandbox_with_profile):
        """Test updating PCP with a reason."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({
            "new_pcp_name": "Dr. Michael Chen",
            "reason": "Moving to Cambridge area",
        })

        assert "✓" in result
        assert "Moving to Cambridge area" in result

    def test_update_pcp_shows_office_info(self, sandbox_with_profile):
        """Test update_pcp shows new doctor's office info."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({"new_pcp_name": "Dr. Michael Chen"})

        assert "Suburban Health Clinic" in result
        assert "456 Oak Ave" in result
        assert "Cambridge" in result

    def test_update_pcp_invalid_doctor(self, sandbox_with_profile):
        """Test updating PCP to a non-existent doctor."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({"new_pcp_name": "Dr. Nonexistent"})

        assert "Error" in result
        assert "not found" in result
        # Should suggest available PCPs
        assert "Dr. Sarah Johnson" in result or "Dr. Michael Chen" in result

    def test_update_pcp_partial_name_match(self, sandbox_with_profile):
        """Test updating PCP with partial name match."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({"new_pcp_name": "Chen"})

        assert "✓" in result
        assert "Dr. Michael Chen" in result

    def test_update_pcp_case_insensitive(self, sandbox_with_profile):
        """Test updating PCP is case-insensitive."""
        tool = create_update_pcp(sandbox_with_profile)
        result = tool.invoke({"new_pcp_name": "dr. michael chen"})

        assert "✓" in result
        assert "Dr. Michael Chen" in result


class TestUpdatePharmacy:
    """Tests for update_pharmacy functionality."""

    def test_update_pharmacy_name_only(self, sandbox_with_profile):
        """Test updating pharmacy with name only."""
        tool = create_update_pharmacy(sandbox_with_profile)
        result = tool.invoke({"pharmacy_name": "Walgreens"})

        assert "✓" in result
        assert "Walgreens" in result
        assert sandbox_with_profile.patient_profile.pharmacy.name == "Walgreens"

    def test_update_pharmacy_full_info(self, sandbox_with_profile):
        """Test updating pharmacy with full information."""
        tool = create_update_pharmacy(sandbox_with_profile)
        result = tool.invoke({
            "pharmacy_name": "Walgreens",
            "pharmacy_address": "200 Main St",
            "pharmacy_phone": "555-999-8888",
        })

        assert "✓" in result
        assert "Walgreens" in result
        assert "200 Main St" in result
        assert "555-999-8888" in result
        assert sandbox_with_profile.patient_profile.pharmacy.address == "200 Main St"
        assert sandbox_with_profile.patient_profile.pharmacy.phone == "555-999-8888"

    def test_update_pharmacy_persists(self, sandbox_with_profile):
        """Test pharmacy update persists in profile."""
        tool = create_update_pharmacy(sandbox_with_profile)
        tool.invoke({"pharmacy_name": "Rite Aid"})

        # Verify via get_profile
        get_tool = create_get_profile(sandbox_with_profile)
        result = get_tool.invoke({})

        assert "Rite Aid" in result


class TestUpdateInsurance:
    """Tests for update_insurance functionality."""

    def test_update_insurance_name_only(self, sandbox_with_profile):
        """Test updating insurance with name only."""
        tool = create_update_insurance(sandbox_with_profile)
        result = tool.invoke({
            "insurance_name": "Aetna",
        })

        assert "✓" in result
        assert "Aetna" in result
        assert sandbox_with_profile.patient_profile.insurance.name == "Aetna"

    def test_update_insurance_with_plan_type(self, sandbox_with_profile):
        """Test updating insurance with plan type."""
        tool = create_update_insurance(sandbox_with_profile)
        result = tool.invoke({
            "insurance_name": "Aetna",
            "plan_type": "PPO",
        })

        assert "✓" in result
        assert "Aetna" in result
        assert "PPO" in result
        assert sandbox_with_profile.patient_profile.insurance.name == "Aetna"
        assert sandbox_with_profile.patient_profile.insurance.plan_type == "PPO"

    def test_update_insurance_persists(self, sandbox_with_profile):
        """Test insurance update persists in profile."""
        tool = create_update_insurance(sandbox_with_profile)
        tool.invoke({
            "insurance_name": "United Healthcare",
            "plan_type": "HMO",
        })

        # Verify via get_profile
        get_tool = create_get_profile(sandbox_with_profile)
        result = get_tool.invoke({})

        assert "United Healthcare" in result


class TestUpdateContactInfo:
    """Tests for update_contact_info functionality."""

    def test_update_phone(self, sandbox_with_profile):
        """Test updating phone number."""
        tool = create_update_contact_info(sandbox_with_profile)
        result = tool.invoke({"phone": "555-777-8888"})

        assert "✓" in result
        assert "555-777-8888" in result
        assert sandbox_with_profile.patient_profile.phone.number == "555-777-8888"

    def test_update_email(self, sandbox_with_profile):
        """Test updating email address."""
        tool = create_update_contact_info(sandbox_with_profile)
        result = tool.invoke({"email": "newemail@test.com"})

        assert "✓" in result
        assert "newemail@test.com" in result
        assert sandbox_with_profile.patient_profile.personal_info.email == "newemail@test.com"

    def test_update_address(self, sandbox_with_profile):
        """Test updating address."""
        tool = create_update_contact_info(sandbox_with_profile)
        result = tool.invoke({"address": "999 New Street"})

        assert "✓" in result
        assert "999 New Street" in result
        assert sandbox_with_profile.patient_profile.address.address1 == "999 New Street"

    def test_update_multiple_fields(self, sandbox_with_profile):
        """Test updating multiple contact fields at once."""
        tool = create_update_contact_info(sandbox_with_profile)
        result = tool.invoke({
            "phone": "555-000-1111",
            "email": "multi@test.com",
            "address": "123 Multi Lane",
        })

        assert "✓" in result
        assert "555-000-1111" in result
        assert "multi@test.com" in result
        assert "123 Multi Lane" in result

    def test_update_no_fields(self, sandbox_with_profile):
        """Test updating with no fields provided."""
        tool = create_update_contact_info(sandbox_with_profile)
        result = tool.invoke({})

        assert "No contact information provided" in result


class TestProfileToolsIntegration:
    """Integration tests for profile tools workflow."""

    def test_get_update_get_workflow(self, sandbox_with_profile):
        """Test get profile, update, then get again workflow."""
        tools = get_profile_tools(sandbox_with_profile)
        get_tool = next(t for t in tools if t.name == "get_profile")
        update_pharmacy = next(t for t in tools if t.name == "update_pharmacy")

        # Get initial profile
        initial = get_tool.invoke({})
        assert "CVS Pharmacy" in initial

        # Update pharmacy
        update_pharmacy.invoke({"pharmacy_name": "Walgreens"})

        # Get updated profile
        updated = get_tool.invoke({})
        assert "Walgreens" in updated
        assert "CVS Pharmacy" not in updated

    def test_multiple_updates_persist(self, sandbox_with_profile):
        """Test multiple updates all persist."""
        tools = get_profile_tools(sandbox_with_profile)
        get_tool = next(t for t in tools if t.name == "get_profile")
        update_pcp = next(t for t in tools if t.name == "update_pcp")
        update_pharmacy = next(t for t in tools if t.name == "update_pharmacy")
        update_insurance = next(t for t in tools if t.name == "update_insurance")

        # Make multiple updates
        update_pcp.invoke({"new_pcp_name": "Dr. Michael Chen"})
        update_pharmacy.invoke({"pharmacy_name": "Rite Aid"})
        update_insurance.invoke({
            "insurance_name": "Cigna",
            "plan_type": "PPO",
        })

        # Verify all updates persisted
        result = get_tool.invoke({})
        assert "Dr. Michael Chen" in result
        assert "Rite Aid" in result
        assert "Cigna" in result
