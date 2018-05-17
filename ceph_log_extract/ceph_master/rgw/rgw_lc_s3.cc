#include <string.h>

#include <iostream>
#include <map>

#include "include/types.h"

#include "rgw_user.h"
#include "rgw_lc_s3.h"


#define dout_subsys ceph_subsys_rgw


bool LCExpiration_S3::xml_end(const char * el) {
  LCDays_S3 *lc_days = static_cast<LCDays_S3 *>(find_first("Days"));
  LCDeleteMarker_S3 *lc_dm = static_cast<LCDeleteMarker_S3 *>(find_first("ExpiredObjectDeleteMarker"));
  LCDate_S3 *lc_date = static_cast<LCDate_S3 *>(find_first("Date"));

  if ((!lc_days && !lc_dm && !lc_date) || (lc_days && lc_dm) 
      || (lc_days && lc_date) || (lc_dm && lc_date)) {
    return false;
  }
  if (lc_days) {
    days = lc_days->get_data();
  } else if (lc_dm) {
    dm_expiration = lc_dm->get_data().compare("true") == 0;
    if (!dm_expiration) {
      return false;
    }
  } else {
    date = lc_date->get_data();
    //We need return xml error according to S3
    boost::optional<ceph::real_time> expiration_date = ceph::from_iso_8601(date);
    if (boost::none == expiration_date) {
      return false;
    }
    struct timespec expiration_time = ceph::real_clock::to_timespec(*expiration_date);
    if (expiration_time.tv_sec % (24*60*60) || expiration_time.tv_nsec) {
      return false;
    }
  }
  return true;
}

bool LCNoncurExpiration_S3::xml_end(const char *el) {
  LCDays_S3 *lc_noncur_days = static_cast<LCDays_S3 *>(find_first("NoncurrentDays"));
  if (!lc_noncur_days) {
    return false;
  }
  days = lc_noncur_days->get_data();
  return true;
}

bool LCMPExpiration_S3::xml_end(const char *el) {
  LCDays_S3 *lc_mp_days = static_cast<LCDays_S3 *>(find_first("DaysAfterInitiation"));
  if (!lc_mp_days) {
    return false;
  }
  days = lc_mp_days->get_data();
  return true;
}

bool RGWLifecycleConfiguration_S3::xml_end(const char *el) {
  XMLObjIter iter = find("Rule");
  LCRule_S3 *rule = static_cast<LCRule_S3 *>(iter.get_next());
  if (!rule)
    return false;
  while (rule) {
    add_rule(rule);
    rule = static_cast<LCRule_S3 *>(iter.get_next());
  }
  return true;
}

bool LCFilter_S3::xml_end(const char* el) {

  XMLObj *o = find_first("And");
  bool single_cond = false;
  int num_conditions = 0;
  // If there is an AND condition, every tag is a child of and
  // else we only support single conditions and return false if we see multiple

  if (o == nullptr){
    o = this;
    single_cond = true;
  }

  RGWXMLDecoder::decode_xml("Prefix", prefix, o);
  if (!prefix.empty())
    num_conditions++;
  auto tags_iter = o->find("Tag");
  obj_tags.clear();
  while (auto tag_xml =tags_iter.get_next()){
    std::string _key,_val;
    RGWXMLDecoder::decode_xml("Key", _key, tag_xml);
    RGWXMLDecoder::decode_xml("Value", _val, tag_xml);
    obj_tags.emplace_tag(std::move(_key), std::move(_val));
    num_conditions++;
  }

  return !(single_cond && num_conditions > 1);
}


bool LCRule_S3::xml_end(const char *el) {
  LCID_S3 *lc_id;
  LCPrefix_S3 *lc_prefix;
  LCStatus_S3 *lc_status;
  LCExpiration_S3 *lc_expiration;
  LCNoncurExpiration_S3 *lc_noncur_expiration;
  LCMPExpiration_S3 *lc_mp_expiration;
  LCFilter_S3 *lc_filter;
  id.clear();
  prefix.clear();
  status.clear();
  dm_expiration = false;

  // S3 generates a 48 bit random ID, maybe we could generate shorter IDs
  static constexpr auto LC_ID_LENGTH = 48;

  lc_id = static_cast<LCID_S3 *>(find_first("ID"));
  if (lc_id){
    id = lc_id->get_data();
  } else {
    gen_rand_alphanumeric_lower(cct, &id, LC_ID_LENGTH);
  }


  lc_filter = static_cast<LCFilter_S3 *>(find_first("Filter"));

  if (lc_filter){
    filter = *lc_filter;
  } else {
    // Ideally the following code should be deprecated and we should return
    // False here, The new S3 LC configuration xml spec. makes Filter mandatory
    // and Prefix optional. However older clients including boto2 still generate
    // xml according to the older spec, where Prefix existed outside of Filter
    // and S3 itself seems to be sloppy on enforcing the mandatory Filter
    // argument. A day will come when S3 enforces their own xml-spec, but it is
    // not this day

    lc_prefix = static_cast<LCPrefix_S3 *>(find_first("Prefix"));

    if (!lc_prefix){
      return false;
    }

    prefix = lc_prefix->get_data();
  }


  lc_status = static_cast<LCStatus_S3 *>(find_first("Status"));
  if (!lc_status)
    return false;
  status = lc_status->get_data();
  if (status.compare("Enabled") != 0 && status.compare("Disabled") != 0)
    return false;

  lc_expiration = static_cast<LCExpiration_S3 *>(find_first("Expiration"));
  lc_noncur_expiration = static_cast<LCNoncurExpiration_S3 *>(find_first("NoncurrentVersionExpiration"));
  lc_mp_expiration = static_cast<LCMPExpiration_S3 *>(find_first("AbortIncompleteMultipartUpload"));
  if (!lc_expiration && !lc_noncur_expiration && !lc_mp_expiration) {
    return false;
  } else {
    if (lc_expiration) {
      if (lc_expiration->has_days()) {
        expiration.set_days(lc_expiration->get_days_str());
      } else if (lc_expiration->has_date()) {
        expiration.set_date(lc_expiration->get_date());
      } else {
        dm_expiration = lc_expiration->get_dm_expiration();
      }
    }
    if (lc_noncur_expiration) {
      noncur_expiration = *lc_noncur_expiration;
    }
    if (lc_mp_expiration) {
      mp_expiration = *lc_mp_expiration;
    }
  }

  return true;
}

void LCRule_S3::to_xml(ostream& out) {
  out << "<Rule>" ;
  out << "<ID>" << id << "</ID>";
  if (!filter.empty()) {
    LCFilter_S3& lc_filter = static_cast<LCFilter_S3&>(filter);
    lc_filter.to_xml(out);
  } else {
    out << "<Prefix>" << prefix << "</Prefix>";
  }
  out << "<Status>" << status << "</Status>";
  if (!expiration.empty() || dm_expiration) {
    LCExpiration_S3 expir(expiration.get_days_str(), expiration.get_date(), dm_expiration);
    expir.to_xml(out);
  }
  if (!noncur_expiration.empty()) {
    LCNoncurExpiration_S3& noncur_expir = static_cast<LCNoncurExpiration_S3&>(noncur_expiration);
    noncur_expir.to_xml(out);
  }
  if (!mp_expiration.empty()) {
    LCMPExpiration_S3& mp_expir = static_cast<LCMPExpiration_S3&>(mp_expiration);
    mp_expir.to_xml(out);
  }
  out << "</Rule>";
}

int RGWLifecycleConfiguration_S3::rebuild(RGWRados *store, RGWLifecycleConfiguration& dest)
{
  int ret = 0;
  multimap<string, LCRule>::iterator iter;
  for (iter = rule_map.begin(); iter != rule_map.end(); ++iter) {
    LCRule& src_rule = iter->second;
    ret = dest.check_and_add_rule(&src_rule);
    if (ret < 0)
      return ret;
  }
  if (!dest.valid()) {
    ret = -ERR_INVALID_REQUEST;
  }
  return ret;
}



void RGWLifecycleConfiguration_S3::dump_xml(Formatter *f) const
{
	f->open_object_section_in_ns("LifecycleConfiguration", XMLNS_AWS_S3);

    for (auto iter = rule_map.begin(); iter != rule_map.end(); ++iter) {
		const LCRule_S3& rule = static_cast<const LCRule_S3&>(iter->second);
		rule.dump_xml(f);
	}

	f->close_section(); // Lifecycle
}

XMLObj *RGWLCXMLParser_S3::alloc_obj(const char *el)
{
  XMLObj * obj = NULL;
  if (strcmp(el, "LifecycleConfiguration") == 0) {
    obj = new RGWLifecycleConfiguration_S3(cct);
  } else if (strcmp(el, "Rule") == 0) {
    obj = new LCRule_S3(cct);
  } else if (strcmp(el, "ID") == 0) {
    obj = new LCID_S3();
  } else if (strcmp(el, "Prefix") == 0) {
    obj = new LCPrefix_S3();
  } else if (strcmp(el, "Filter") == 0) {
    obj = new LCFilter_S3();
  } else if (strcmp(el, "Status") == 0) {
    obj = new LCStatus_S3();
  } else if (strcmp(el, "Expiration") == 0) {
    obj = new LCExpiration_S3();
  } else if (strcmp(el, "Days") == 0) {
    obj = new LCDays_S3();
  } else if (strcmp(el, "Date") == 0) {
    obj = new LCDate_S3();
  } else if (strcmp(el, "ExpiredObjectDeleteMarker") == 0) {
    obj = new LCDeleteMarker_S3();
  } else if (strcmp(el, "NoncurrentVersionExpiration") == 0) {
    obj = new LCNoncurExpiration_S3();
  } else if (strcmp(el, "NoncurrentDays") == 0) {
    obj = new LCDays_S3();
  } else if (strcmp(el, "AbortIncompleteMultipartUpload") == 0) {
    obj = new LCMPExpiration_S3();
  } else if (strcmp(el, "DaysAfterInitiation") == 0) {
    obj = new LCDays_S3();
  }
  return obj;
}
